<?php

ini_set("max_execution_time", "-1");
ini_set('memory_limit','-1');
set_time_limit(0);

include('Functions.php');

// Where are the images and labels
$train_directory = __DIR__ . DIRECTORY_SEPARATOR . 'Training Images' . DIRECTORY_SEPARATOR . 'train';
$test_directory = __DIR__ . DIRECTORY_SEPARATOR .  'Training Images' .  DIRECTORY_SEPARATOR . 'test';


// Where is the data
$train_save_directory = __DIR__ . DIRECTORY_SEPARATOR . 'Training Data';
$test_save_directory = __DIR__ . DIRECTORY_SEPARATOR .  'Training Data';


// Convolution_kernels we might want to use
$convolution_kernels = array(
    // This is just a pass-through of the original image
    'identity'=>array(array(0, 0, 0), 
                      array(0, 1, 0), 
                      array(0, 0, 0)
                     ),
    // Shift kernels - basically... these shift the image 1 pixel in the specified direction
    'shift_north_west'=>array(array(4, 0, 0),
                              array(0, -1, 0),
                              array(0, 0, -4)
                             ),
    'shift_north'=>array(array(0, 4, 0),
                         array(0, -1, 0),
                         array(0, -4, 0)
                        ),
    'shift_north_east'=>array(array(0, 0, 4),
                              array(0, -1, 0),
                              array(-4, 0, 0)
                             ),
    'shift_east'=>array(array(0, 0, 0),
                        array(-4, -1, 4),
                        array(0, 0, 0)
                       ),
    'shift_south_east'=>array(array(-4, 0, 0),
                              array(0, -1, 0),
                              array(0, 0, 4)
                             ),
    'shift_south'=>array(array(0, -4, 0),
                         array(0, -1, 0),
                         array(0, 4, 0)
                        ),
    'shift_south_west'=>array(array(0, 0, -4),
                              array(0, -1, 0),
                              array(4, 0, 0)
                             ),
    'shift_west'=>array(array(0, 0, 0),
                        array(4, -1, -4),
                        array(0, 0, 0)
                       ),
    // Emboss kernels - These highlight and shadow angles and boundaries https://en.wikipedia.org/wiki/Image_embossing
    'emboss_1'=>array(array(-2, -1, 0),
                      array(-1, 0, 1),
                      array(0, 1, 2)
                     ),
    'emboss_2'=>array(array(0, -1, -2),
                      array(1, 0, -1),
                      array(2, 1, 0)
                     ),
    'emboss_3'=>array(array(2, 1, 0),
                      array(1, 0, -1),
                      array(0, -1, -2)
                     ),
    'emboss_4'=>array(array(0, 1, 2), 
                      array(-1, 0, 1),
                      array(-2, -1, 0)
                     ),
    // Edges - Find the edges and outline - https://en.wikipedia.org/wiki/Kernel_(image_processing)
    'edge_outline'=>array(array(0, 1, 0),
                          array(1, -4, 1),
                          array(0, 1, 0)
                         ),
    'edge_horizontal'=>array(array(-1, -1, -1),
                             array(2, 2, 2),
                             array(-1, -1, -1)
                            ),
    'edge_vertical'=>array(array(-1, 2, -1),
                           array(-1, 2, -1),
                           array(-1, 2, -1)
                           ),
    // Area kernel
    'area'=>array(array(-4, 4, -4),
                  array(4, -4, 4),
                  array(-4, 4, -4)
                 )
); // / convolution_kernels


// List of Layers
// Note that Flattening should always be last
$layers = array(CONVOLUTION_LAYER, 
                POOLING_LAYER, 
                //CONVOLUTION_LAYER, 
                //POOLING_LAYER, 
                FLATTENING_LAYER); // Always flatten, and always last
                


// List of kernels we will actually use
// Uncomment the ones you want but each one adds to the size 
// of the ANN input layer and increases the size of the training 
// data. Additionally, it will result in slower training, however
// experiment with different kernels and your dataset to see which 
// (if any) work best with your dataset.
$kernels = array(
                 'identity'=>$convolution_kernels['identity'],
                 //'edge_outline'=>$convolution_kernels['edge_outline'],
                 //'edge_horizontal'=>$convolution_kernels['edge_horizontal'], 
                 //'edge_vertical'=>$convolution_kernels['edge_vertical'], 
                 'area'=>$convolution_kernels['area']
                 //'emboss_1'=>$convolution_kernels['emboss_1'],
                 //'emboss_2'=>$convolution_kernels['emboss_2'],
                 //'emboss_3'=>$convolution_kernels['emboss_3'],
                 //'emboss_4'=>$convolution_kernels['emboss_4'],
                 //'shift_north_west'=>$convolution_kernels['shift_north_west'],
                 //'shift_north'=>$convolution_kernels['shift_north'],
                 //'shift_north_east'=>$convolution_kernels['shift_north_east'],
                 //'shift_east'=>$convolution_kernels['shift_east'],
                 //'shift_south_east'=>$convolution_kernels['shift_south_east'],
                 //'shift_south'=>$convolution_kernels['shift_south'],
                 //'shift_south_west'=>$convolution_kernels['shift_south_west'],
                 //'shift_west'=>$convolution_kernels['shift_west']
                );



// Pool size - We use only 1 pooling layer before flattening
// The pooling function and as written, it isn't that robust, 
// meaning that it doesn't handle empty pixels on the edge at all, 
// this is because currently it divides the image into equal sized groups
// that must line up inside the pool matrix completely. The pooling function 
// will attempt to adjust the matrix size for you in one direction (up) if 
// the size selected results in a bad configuration.
// 
// Adding an adjustable stride (not that difficult at all) and selecting and implementing 
// an empty pixel strategy to handle matrix grid sizes that result in empty pixels  (moderately difficult)
// Would greatly improve this implementation and would probably be one of the first things on the to do
// list.
// 
// So... given the aforementioned... pooling the MNIST dataset 
// at the minimum pool size of 2 results in only being possible to pool a maximum of
// 2 times, there are alternatives to adding stride... but add pooling stride!
$pooling_size = 2; // This is the number of pixels
                   // in 1D of our 2D pooling matrix
                   //
                   // More pixels per pool means a smaller 
                   // output pooled image because more pixels
                   // are pooled into fewer pixels.
                   //
                   // e.g. using an image that is 28 x 28 (like the MNIST image set)
                   // 
                   // 2 = 28 / 2x2 matrix = 14 - 14x14 pooled image (196 input neurons required)
                   // 4 = 28 / 4x4 matrix = 7 - 7x7 pooled image    (49 input neurons required)
                   // 7 = 28 / 7x7 matrix = 4 - 4x4 pooled image    (16 input neurons required)
                   // 14 = 28 / 14x14 matrix = 2 - 2x2 pooled image (4 input neurons required)

$pooling_method = MAX_POOL;// Options:
                           // MAX_POOL
                           // MIN_POOL
                           // AVG_POOL

echo 'Generating training and test data data from labels and images...' . PHP_EOL;

// Generate  minst.train.data file from the images and labels
GenerateDatasetFromLabeledImages($train_directory, 
                                'minst_train_labels.txt',
                                $train_save_directory,
                                'minst.train.data', 
                                $layers,
                                $kernels, 
                                $pooling_method, 
                                $pooling_size);

// Generate  minst.test.data file from the images and labels    
GenerateDatasetFromLabeledImages($test_directory, 
                                'minst_test_labels.txt',
                                $test_save_directory,                                
                                'minst.test.data', 
                                $layers, 
                                $kernels, 
                                $pooling_method, 
                                $pooling_size);

echo 'All Done!' . PHP_EOL;