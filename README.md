# HandWriting Recognition

## Data

[MNIST](http://yann.lecun.com/exdb/mnist/)

## Tech

CNN

`for accuracy(mark) I have to choose tensorflow, but I'd like to implement it by myself later when free`

## Env

- Python 3.6.2 (conda)
- CUDA 10.0
- cuDNN 7.4
- tensorflow-gpu-1.14.0
- keras
- jupyter notebook (optional)

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

```python
import struct
import numpy as np

# idx1 is likely
def parse_idx3(path):
    # read bin data from file
    bin_data = open(path, 'rb').read()

    # parse head info
    offset = 0
    # read first 4 line
    format_head = '>iiii'
    magic_number, item_num, row_num, col_num = struct.unpack_from(format_head, bin_data, offset)
    # show magic number and dimensions size
    file_name = path.split('/')[-1].split('.')[0]
    print(('='*15) + file_name + ('='*15))
    print('magic number: %d' % magic_number)
    print('item num: %d, %d row * %d col' % (item_num, row_num, col_num))
    print('=' * (30 + len(file_name)))

    # parse data
    item_size = row_num * col_num
    # read one item once
    format_data = '>' + str(item_size) + 'B'
    # set offest to where we have finished reading
    offset = struct.calcsize(format_head)
    # create an empty array and fill it
    items = np.empty((item_num, row_num, col_num))
    for i in range(item_num):
        items[i] = np.array(struct.unpack_from(format_data, bin_data, offset)).reshape((row_num, col_num))
        offset = offset + struct.calcsize(format_data)
    return items
```
it only means read the head then info and read data according to it

it is a simple func, and it will return an multidimensional array

func to parse idx1 is the same

```python
def parse_idx1(path):
    bin_data = open(path, 'rb').read()
    offset = 0
    format_head = '>ii'
    magic_number, item_num = struct.unpack_from(format_head, bin_data, offset)
    file_name = path.split('/')[-1].split('.')[0]
    print(('='*15) + file_name + ('='*15))
    print('magic number: %d' % magic_number)
    print('item num: %d' % item_num)
    print('=' * (30 + len(file_name)))
    format_data = '>B'
    offset = struct.calcsize(format_head)
    items = np.empty(item_num)
    for i in range(item_num):
        items[i] = struct.unpack_from(format_data, bin_data, offset)[0]
        offset = offset + struct.calcsize(format_data)
    return items
```

### Process Data

#### One-Hot Encoding

- solution: change n types feature to n-length arrays
  - [0,1,2] => [[1,0,0], [0,1,0], [0,0,1]]
- meaning: make distance calculation more reasonable

```python
from keras.utils import to_categorical

train_labels = parse_idx1(train_labels_path)
train_labels = to_categorical(train_labels, 10)
```

#### Formatting

- solution: [[28x28]] => [[[1x28]x28]]
  - [[[1,...,0],...,[0,...,0]]] => [[[[1],...,[0]],...,[[0],...[0]]]]
- meaning: satisfy model requirement

```python
train_images = parse_idx3(train_images_path)
train_images = train_images.reshape(-1, 28, 28, 1)
train_images = train_images.astype('float32')
train_images /= 255
```
remember do the same thing to the test set !

### Build Model

not much to say, just see the code and try to optimize it

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

model = Sequential()
# Conv Layer
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
model.add(Conv2D(64, (5,5), activation='relu'))
# Pool Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Flatten Layer
model.add(Flatten())
# Dropout Layer
model.add(Dropout(0.5))
# Dense Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# compile model (set loss and optimize func)
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

# data num in one batch
batch_size = 100
# times to cover all training data
epochs = 9
# start training
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)

# test accuracy
loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print('loss: %.4f accuracy: %.4f' %(loss, accuracy))
```

- Layer
  - Conv Layer: extract feature
  - Pool Layer: reduce feature dimension
  - Flatten Layer: transfer data to one dimension (dont affect batch size)
  - Dropout Layer: randomly drop some input
  - Dense Layer: full connection network
- attr
  - batch_size: n data => 1 batch, reduce iteration time
  - epochs: iteration times

now run the code and see its accuracy

if you are not satisfied, you can change layer structure and attrs to improve it

> the latest code will not be updated in this .md file, you can get it in .ipynb file, but I will show you the latest accuracy

latest accuracy: `99.20%`
