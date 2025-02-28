import unittest

from src.dataset.snow import SnowDataset


class TestSnowDataset(unittest.TestCase):
    def test_get_item(self) -> None:
        snow_dataset = SnowDataset(
            dataset_dir_path='/Users/samuelthomas/Documents/University/4thYr_Final'
                             '/ECM3401_Individual_Literature_Review_and_Project/SNOW_Semantic_Segmentation'
                             '/snow_dataset',
            resize=True,
            normalize=True,
            rotate=True,
            augment_image=True,
        )

        image_1, mask_1 = snow_dataset[0]
        self.assertTrue(type(image_1).__name__ == 'Tensor')
        self.assertTrue(type(mask_1).__name__ == 'Tensor')
        self.assertEqual(image_1.size(), (3, 256, 256))
        self.assertEqual(mask_1.size(), (1, 256, 256))

        image_2, mask_2 = snow_dataset[1]
        self.assertTrue(type(image_2).__name__ == 'Tensor')
        self.assertTrue(type(mask_2).__name__ == 'Tensor')
        self.assertEqual(image_2.size(), (3, 256, 256))
        self.assertEqual(mask_2.size(), (1, 256, 256))


if __name__ == '__main__':
    unittest.main()
