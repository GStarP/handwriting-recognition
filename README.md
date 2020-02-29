# HandWriting Recognition

## Data

[MNIST](http://yann.lecun.com/exdb/mnist/)

## Tech

CNN

## Step

### Read Data

> How to read data from .idx(n)-ubyte ?

IDX is for multidimensional array, idx(n) means x-dimension(s)

first, let's see its format

```
magic number
dimension1 size
...
dimensionn size
data
```

- magic number
  - first 2 bytes: always 0
  - 3rd byte: data type
    - 0x08: unsigned byte
    - 0x09: singned byte
    - 0x0B: short  (2 bytes)
    - 0x0C: int    (4 bytes)
    - 0x0D: float  (4 bytes)
    - 0x0E: double (8 bytes)
  - 4th byte: dimension num

now see the parse function (Pyhton)

