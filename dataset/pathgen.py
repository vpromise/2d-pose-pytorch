import os

# save path
save_path = './dataset/test'

if os.path.isdir(save_path):
    pass
else:
    os.makedirs(save_path)

train_path = save_path + '/train.txt'
valid_path = save_path + '/valid.txt'
test_path = save_path + '/test.txt'

# data path
data = '/media/vpromise/Inter/orthopedic/2d_data/'
img, hm = 'img', 'coor'

# generate path txt

with open(train_path, 'w') as train_f:
    for i in range(1, 181):
        for j in range(0, 400,400):
            train_line = str(data + img + '/S' + str('%03d' % i) + '/left/' + str('%04d' % j) + '.jpg' +
                             ',' + data + hm + '/S' + str('%03d' % i) + '/left/' + str('%04d' % j) + '.npy\r\n')
            train_f.write(train_line)
    print('Successfully created train files at ', train_path)

with open(valid_path, 'w') as valid_f:
    for i in range(181, 201):
        for j in range(0, 400,400):
            valid_line = str(data + img + '/S' + str('%03d' % i) + '/left/' + str('%04d' % j) + '.jpg' +
                             ',' + data + hm + '/S' + str('%03d' % i) + '/left/' + str('%04d' % j) + '.npy\r\n')
            valid_f.write(valid_line)
    print('Successfully created valid files at ', valid_path)


with open(test_path, 'w') as test_f:
    for i in range(201, 226):
        for j in range(0, 400):
            test_line = str(data + img + '/S' + str('%03d'%i) + '/left/' + str('%04d'%j) + '.jpg' + 
                             ',' + data + hm + '/S' + str('%03d' % i) + '/left/' + str('%04d' % j) + '.npy\r\n')
            test_f.write(test_line)
    print('Successfully created test files at ', test_path)