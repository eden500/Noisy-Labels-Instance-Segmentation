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


## Citation


If you use this benchmark in your research, please cite this project.


```
Bibtex will be avalible shortly
```


## License

This project is released under the [Apache 2.0 license](https://github.com/eden500/Noisy-Labels-Instance-Segmentation/blob/main/LICENSE.txt).


Please make sure you use it with proper licenced Datasets.

We use [MS-COCO/LVIS](https://cocodataset.org/#termsofuse) and [Cityscapes](https://www.cityscapes-dataset.com/license/)


![image](https://github.com/eden500/Noisy-Labels-Instance-Segmentation/assets/66938362/3e22ad79-3f12-4767-b994-2df57dd265e7)
