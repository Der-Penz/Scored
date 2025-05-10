from typing import List
import numpy as np
from scored.data_preparation.annotation import LabelStudioAnnotation, LabelStudioKeypoint, LabelStudioObject
import imgaug.augmentables.kps as ia_kps
import imgaug.augmenters as iaa

from scored.util import compute_bounding_box


def augment_image(image : np.ndarray, annotation: LabelStudioAnnotation, augmentations : iaa.Augmenter):
    """
    Apply a series of augmentations to an image.

    Args:
        image (numpy.ndarray): The input image to augment.
        annotation (LabelStudioAnnotation): The annotation object containing information about the image.
        augmentations (iaa.Augmenter): Augmentations to apply.

    Returns:
        Augmented image and the augmented annotation.
    """

    all_keypoints = []
    keypoints_by_object = []

    for obj in annotation.objects:
        if obj.keypoints:
            obj_keypoints = [ia_kps.Keypoint(x=kp.pos[0] * image.shape[0], y=kp.pos[1] * image.shape[1]) for kp in obj.keypoints]
            keypoints_by_object.append(obj_keypoints)
            all_keypoints.extend(obj_keypoints)
        else:
            keypoints_by_object.append([])

    kps_on_image = ia_kps.KeypointsOnImage(all_keypoints, shape=image.shape)

    image_aug, kps_aug = augmentations(image=image, keypoints=kps_on_image)


    augmented_objects = []

    start_idx = 0
    for obj in annotation.objects:
        if obj.keypoints:
            augmented_keypoints : List[LabelStudioKeypoint] = []
            for i, key in enumerate(kps_aug.keypoints[start_idx:start_idx + len(obj.keypoints)]):
                augmented_keypoints.append(
                    LabelStudioKeypoint(
                        pos=(key.x / image_aug.shape[0], key.y / image_aug.shape[1]),
                        label=obj.keypoints[i].label,
                        id=obj.keypoints[i].id,
                        parent_id=obj.keypoints[i].parent_id,
                    )
                )

            bb = compute_bounding_box(
                key_points=[(key.pos[0], key.pos[1]) for key in augmented_keypoints]
            )

            augmented_objects.append(LabelStudioObject(
                label=obj.label,
                id=obj.id,
                bbox=bb,
                keypoints=augmented_keypoints,
            ))

            start_idx += len(obj.keypoints)
            
        else:
            augmented_objects.append(obj)

    annotation_aug = LabelStudioAnnotation(
        id=annotation.id,
        img_path=annotation.img_path,
        objects=augmented_objects,
    )
    return image_aug, annotation_aug