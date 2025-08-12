from typing import Optional

class NoFaceDetectionException (Exception):
    pass

class RoiExtractor:
    left_eye_id = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye_id = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mouth_id = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]

    def __init__(self):
        from mediapipe import solutions
        self.face_mesh = solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmark (self, image):
        from mediapipe import solutions
        from numpy import zeros, float32

        width = image.shape[1]
        height = image.shape[0]

        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            raise NoFaceDetectionException()

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [l for l in face_landmarks.landmark]

        ldmks = zeros((468, 5), dtype=float32)
        ldmks[:, 0] = -1.0
        ldmks[:, 1] = -1.0

        for idx in range(len(landmarks)):
            landmark = landmarks[idx]
            if not (
                (landmark.HasField('visibility') and landmark.visibility < 0.5) or \
                (landmark.HasField('presence') and landmark.presence < 0.5)
            ):
                coords = solutions.drawing_utils._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, width, height)
                if coords:
                    ldmks[idx, 0] = coords[1]
                    ldmks[idx, 1] = coords[0]

        return ldmks

    def extract_holistic_mask (self, image, ldmks):
        from cupy import asarray as as_cupy_array
        image = as_cupy_array(image)

        full_face_mask = as_cupy_array(
            self.__create_mask(ldmks, image.shape, None)
        )
        left_eye_mask = as_cupy_array(
            self.__create_mask(ldmks, image.shape, self.left_eye_id)
        )
        right_eye_mask = as_cupy_array(
            self.__create_mask(ldmks, image.shape, self.right_eye_id)
        )
        mounth_mask = as_cupy_array(
            self.__create_mask(ldmks, image.shape, self.mouth_id)
        )

        mask = full_face_mask * (1-left_eye_mask) * (1-right_eye_mask) * (1-mounth_mask)

        return mask, image * mask

    def __create_mask (self, ldmks, image_shape, roi: Optional[list]):
        """
        :param roi: should be None or one of MagicLandmarks from pyVHR.extraction.sig_processing.MagicLandmarks
        """
        from numpy import array, expand_dims
        from scipy.spatial import ConvexHull
        from PIL import Image, ImageDraw

        if roi is None:
            roi_ldmks = ldmks
        else:
            roi_ldmks = ldmks[roi]

        aviable_ldmks = roi_ldmks[roi_ldmks[:, 0] >= 0][:, :2]
        assert len(aviable_ldmks) > 3

        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v, 0], aviable_ldmks[v, 1]) for v in hull.vertices]
        img = Image.new('L', image_shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        left_eye_mask = array(img)
        return expand_dims(left_eye_mask, axis=0).T

if __name__ == "__main__":
    import scipy.io as sio
    from numpy import array, uint8, full, nan
    from tqdm import tqdm
    from os.path import join
    from PIL import Image
    from pickle import dump

    extractor = RoiExtractor()

    save_folder_path = ""
    origin_folder_path = join(save_folder_path, 'origin')
    masked_folder_path = join(save_folder_path, 'masked')
    mask_folder_path = join(save_folder_path, 'mask')

    f = sio.loadmat('p1_0.mat')
    frame_shape = array(f['video'][0]).shape

    maskes = full(
        (len(f['video']), frame_shape[0], frame_shape[1], 1),
        nan
    )
    for idx, frame in enumerate(tqdm(f['video'])):
        image = array(frame) * 255
        image = image.astype(uint8)

        Image.fromarray(image).save(join(origin_folder_path, f"{idx + 1}.png"))

        try:
            ldmks = extractor.extract_landmark(image)
            mask, masked_image = extractor.extract_holistic_mask(image, ldmks)
            maskes[idx] = mask.get()

            Image.fromarray(masked_image.get()).save(join(masked_folder_path, f"{idx + 1}.png"))

            mask_reshape = mask.reshape(mask.shape[0], mask.shape[1])
            Image.fromarray((mask_reshape * 255).astype('uint8').get(), 'L').save(join(mask_folder_path, f"{idx + 1}.png"))
        except NoFaceDetectionException:
            pass

    with open(join(save_folder_path, "mask.pkl"), 'wb') as f:
        dump(maskes, f)
