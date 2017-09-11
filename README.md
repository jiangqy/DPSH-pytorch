---
Source code for paper "Feature Learning based Deep Supervised Hashing with Pairwise Labels"
---
### 1. Running example:
Environment: python 3

Requirements:
```python
pytorch
torchvision
```
### 2. State:
As pytorch doesn't provide pretrained VGG-F model, unlike original DPSH [paper](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf), we use pretrained Alexnet or pretrained VGG-11 for feature learning part instead of pretrained VGG-F.
### 3. Data processing:
### 4. Result:
<table>
    <tr>
        <td rowspan="2">Net Structure</td>    
        <td colspan="4">Code Length</td>
    </tr>
    <tr>
        <td >12 bits</td><td >24 bits</td> <td >32 bits</td><td >48 bits</td>  
    </tr>
    <tr>
        <td >VGG-F[^hello]</td><td >24 bits</td> <td >32 bits</td><td >48 bits</td>  
    </tr>
</table>