# RegNet

<p align="center">
  <img src="https://github.com/iVishalr/RegNetX/blob/main/doc/designspace.png" alt="Designing Network Design Spaces" />
</p>

Pytorch Implementation of "Desigining Network Design Spaces", Radosavovic et al. CVPR 2020. 

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf) | [Official Implementation](https://github.com/facebookresearch/pycls)

RegNet offer a very nice design space for neural network architectures. RegNet design space consists of networks with simple structure which authors call "Regular" Networks (RegNet). Models in RegNet design space have higher concentration of models that perform well and generalise well. RegNet models are very efficient and run upto 5 times faster than EfficientNet models on GPUs.

Also RegNet models have been used as a backbone in Tesla FSD Stack.

## Overview Of AnyNet

- Main goal of the paper is to help in better understanding of network design and discover principles that generalize across settings.
- Explore structure aspeck of network design and arrive at low dimensional design space consisting of simple regualar networks
- Network width and depth can be explained by a quantized linear function.

### AnyNet Design Space

The basic structure of models in AnyNet design space consists of a simple Stem which is then followed by the network body that does majority of the computation and a final network head that predicts the class scores. The stem and head networks are kept as simple as possible. The network body consists of 4 stages that operate at progressively lower resolutions.

<p align="center">
  <img src="https://github.com/iVishalr/RegNetX/blob/main/doc/anynet.png" alt="AnyNet" />
</p>

Structure of network body is determined by block width `w`, network depth `d_i`, bottleneck ratio `b_i` and group widths `g`. Degrees of freedom at stage 'i' are number of blocks `d` in each stage, block width `w` and other block parameters such as stride, padding and so on.

Other models are obtained by refining the design space by adding more constraints on the above parameters. Design space is refined keeping the following things in mind :
- Simplify structure of design space.
- Improve the interpretability of design space.
- Maintain Design space complexity.
- Maintain model diversity in design space.

### AnyNetX

<p align="center">
  <img src="https://github.com/iVishalr/RegNetX/blob/main/doc/xblock.png" alt="XBlock" />
</p>


- Uses XBlocks within each block of the network
- Degrees of freedom in AnyNetX is 16
- Each network has 4 stages
- Each stage has 4 parameters (network depth di, block width wi, bottleneck ratio bi, group width gi)
- bi ∈ {1,2,4}
- gi ∈ {1,2,3,...,32}
- wi <= 1024
- di <= 16

### AnyNetX(A)

AnyNetX(A) is same as the above AnyNetX

### AnyNetX(B)

In this design space, 
- bottleneck ratio bi is fixed for all stages.
- performance of models in AnyNetX(B) space is almost equal to AnyNetX(A) in average and best case senarios
- bi <= 2 seemes to work best.

### AnyNetX(C)

In this design space,
- Shared group width gi for all stages.
- AnyNetX(C) has 6 fewer degrees of freedom compared to AnyNetX(A)
- gi > 1 seems to work best

### AnyNetX(D)

In AnyNetX(D) design space, authors observed that good networks have increasing stage widths w(i+1) > wi

### AnyNetX(E) 

In AnyNetX(E) design space, it was observed that as stage widths wi increases, depth di likewise tend to increase except for the last stage.

### RegNet

Please refer to Section 3.3 in paper.

# References

```
@InProceedings{Radosavovic2020,
  title = {Designing Network Design Spaces},
  author = {Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Doll{\'a}r},
  booktitle = {CVPR},
  year = {2020}
}
```

# LICENSE

MIT
