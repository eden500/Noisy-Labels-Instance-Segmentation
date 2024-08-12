# Noisy-Labels-Instance-Segmentation
## This is the official repo for the paper A Benchmark for Learning with Noisy Labels in Instance Segmentation

![paper meme](https://github.com/eden500/Noisy-Labels-Instance-Segmentation/assets/66938362/e786140b-cd28-41d3-8193-2529f1ed37d5)

### ReadMe:
Important! The original annotations should be in coco format.

To run the benchmark, run the following:
```
python noise_annotations.py /path/to/annotations --benchmark {easy, medium, hard} (choose the benchmark level) --seed 1
```

For example:
```
python noise_annotations.py /path/to/annotations --benchmark easy --seed 1
```


To run a custom noise method, run the following:
```
python noise_annotations.py /path/to/annotations --method_name method_name --corruption_values [{'rand': [scale_proportion, kernel_size(should be odd number)],'localization': [scale_proportion, std_dev], 'approximation': [scale_proportion, tolerance], 'flip_class': percent_class_noise}]}]
```

For example:
```
 python noise_annotations.py /path/to/annotations --method_name my_noise_method --corruption_values [{'rand': [0.2, 3], 'localization': [0.2, 2], 'approximation': [0.2, 5], 'flip_class': 0.2}]
```

To run the coco-wan, do the following:

download the weights for sam vit-h:
```
mkdir -p weights
cd weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Create a conda env with the required libraries:
```
 conda env create -f wan_environment.yml

 conda activate cuda_env
```

For the easy noise, run create_gt_plus_sam_noise.py:
```
python create_gt_plus_sam_noise.py --annotations_path=a_path --data_path=d_path --sam_path=s_path
```

For example:
```
python create_gt_plus_sam_noise.py --annotations_path=data/coco_ann2017/annotations --data_path=data --sam_path=weights/sam_vit_h_4b8939.pth
```

For the medium noise, do the exact same thing but with the file create_gt_plus_sam_point_noise.py


For the hard noise, apply class noise on top of the medium weak annotation noise by running noise_annotations.py with the proper arguments.



## Citation


If you use this benchmark in your research, please cite this project.


```
@misc{grad2024benchmarkinglabelnoiseinstance,
      title={Benchmarking Label Noise in Instance Segmentation: Spatial Noise Matters}, 
      author={Eden Grad and Moshe Kimhi and Lion Halika and Chaim Baskin},
      year={2024},
      eprint={2406.10891},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.10891}, 
}
```


## License

This project is released under the [Apache 2.0 license](https://github.com/eden500/Noisy-Labels-Instance-Segmentation/blob/main/LICENSE.txt).


Please make sure you use it with proper licenced Datasets.

We use [MS-COCO/LVIS](https://cocodataset.org/#termsofuse) and [Cityscapes](https://www.cityscapes-dataset.com/license/)


![image](https://github.com/eden500/Noisy-Labels-Instance-Segmentation/assets/66938362/3e22ad79-3f12-4767-b994-2df57dd265e7)
