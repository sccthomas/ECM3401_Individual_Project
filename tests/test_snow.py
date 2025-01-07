import unittest

from dataset.snow import SnowDataset


class MyTestCase(unittest.TestCase):
    def test_something(self):
        snow_dataset = SnowDataset(
            dataset_dir_path='/Users/samuelthomas/Documents/University/4thYr_Final'
                             '/ECM3401_Individual_Literature_Review_and_Project/SNOW_Semantic_Segmentation'
                             '/snow_dataset')

        image, mask = snow_dataset[0]
        self.assertEqual(image.size(), (3, 512, 512))
        self.assertEqual(mask.size(), (1, 512, 512))


if __name__ == '__main__':
    unittest.main()
