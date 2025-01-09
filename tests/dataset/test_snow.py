import unittest

from src.dataset.snow import SnowDataset


class MyTestCase(unittest.TestCase):
    def test_something(self):
        snow_dataset = SnowDataset(
            dataset_dir_path='/Users/samuelthomas/Documents/University/4thYr_Final'
                             '/ECM3401_Individual_Literature_Review_and_Project/SNOW_Semantic_Segmentation'
                             '/snow_dataset')

        image, mask = snow_dataset[0]
        self.assertTrue(type(image).__name__ == 'Tensor')
        self.assertTrue(type(mask).__name__ == 'Tensor')
        self.assertEqual(image.size(), (3, 256, 256))
        self.assertEqual(mask.size(), (1, 256, 256))


if __name__ == '__main__':
    unittest.main()
