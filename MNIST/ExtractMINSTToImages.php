<?php


/*

This code requires a 64 Bit PHP installation and if you see an error like this:

VirtualAlloc() failed: [0x00000008] Not enough memory resources are available to process this command.


VirtualAlloc() failed: [0x00000008] Not enough memory resources are available to process this command.

PHP Fatal error:  Out of memory (allocated 813694976) (tried to allocate 805306376 bytes) in ... ExtractMINSTToImages.php on line 20

Install PHP 64 Bit: 

https://www.php.net/downloads
https://windows.php.net/download/

Instead of reading/processing the entire database file into memory...

processing the data image by image would make it possible for a 32 bit OS & PHP instillation 

to extract the images but I got lazy and just did it all in memory. :-P

*/


ini_set("max_execution_time", "-1");
ini_set('memory_limit','-1');
set_time_limit(0);


/////////////////////////////////////////////////////////////////////////////////
// Functions

function GetFileBytes($file_path){
    
    $file_handle = fopen($file_path, 'r'); // Open File

    $raw_bytes = array(); // Raw bytes will be read 1 at a time into this array
                          // until the entire file has been read
                          
    while (!feof($file_handle)) { // From now till the end of the file
            $raw_bytes[] = fread($file_handle, 1); // Read 1 Byte of data - EOF
    }
    
    fclose($file_handle); // Close File
    
    return $raw_bytes; // Return raw byte data
}


function ExtractLabelFromByteData($labels_file, $save_directory, $save_file){
    
    $raw_bytes = GetFileBytes($labels_file);
    
    // magic number (MSB first) - Bytes 0-3 (32 bits)
    $magic_number = $raw_bytes[0]
                    . $raw_bytes[1]
                    . $raw_bytes[2]
                    . $raw_bytes[3];
                    
    // number of items - Bytes 4-7 (32 bits)
    $number_of_items = unpack('N', $raw_bytes[4]
                    . $raw_bytes[5]
                    . $raw_bytes[6]
                    . $raw_bytes[7]); // 32 bit Unsigned Int
    $number_of_items = $number_of_items[1];

    $labels = array();
    $curr_label = 0;
    for($bit = 8; $bit < $number_of_items + 8; $bit++){
        $b1 = $raw_bytes[$bit];
        $label = unpack('C', $b1);
        $labels[] = "$curr_label.png " . $label[1];
        $curr_label++;
    }
        
    // Create Labels File
    $unpacked_labels_file = fopen($save_directory . DIRECTORY_SEPARATOR . $save_file, 'w');
    fwrite($unpacked_labels_file, implode(PHP_EOL, $labels)); // Write Labels
    fclose($unpacked_labels_file); // Close File
    
    // Free memory
    $raw_bytes = NULL;
    unset($raw_bytes);
}


function ExtractImagesFromByteData($images_file, $save_directory){
    
    // Open File and Get Bytes
    $raw_bytes = GetFileBytes($images_file);
    
    // magic number (MSB first) - Bytes 0-3 (32 bits)
    $magic_number = $raw_bytes[0]
                    . $raw_bytes[1]
                    . $raw_bytes[2]
                    . $raw_bytes[3];
                    
    // number of images - Bytes 4-7 (32 bits)
    $number_of_images = unpack('N', $raw_bytes[4]
                    . $raw_bytes[5]
                    . $raw_bytes[6]
                    . $raw_bytes[7]); // 32 bit Unsigned Int
    $number_of_images = $number_of_images[1];
    
    
    // number of rows - Bytes 8-11 (32 bits)
    $number_of_rows = unpack('N', $raw_bytes[8]
                    . $raw_bytes[9]
                    . $raw_bytes[10]
                    . $raw_bytes[11]); // 32 bit Unsigned Int
                                    
    $number_of_rows = $number_of_rows[1];
    
    // number of columns - Bytes 12-15 (32 bits)
    $number_of_columns = unpack('N', $raw_bytes[12]
                    . $raw_bytes[13]
                    . $raw_bytes[14]
                    . $raw_bytes[15]); // 32 bit Unsigned Int
    $number_of_columns = $number_of_columns[1];

    $bytes_per_image = $number_of_rows * $number_of_columns;


    $current_bit = 16;
    for($curr_image = 0; $curr_image < $number_of_images; $curr_image++){
        $pixels = array();
        for($bit = 0; $bit < $bytes_per_image; $bit++){
            $pixel = unpack('C', $raw_bytes[$current_bit]);
            $pixels[] = $pixel[1];
            $current_bit++;
        }

        $im = imagecreate($number_of_columns, $number_of_rows);

        // Sets background to black
        $background = imagecolorallocate($im, 0, 0, 0);

        // Allocate colors
        $white = imagecolorallocate($im, 255, 255, 255);
        $black = imagecolorallocate($im, 0, 0, 0);
        
        $curr_pixel = 0;
        for($row = 0; $row < $number_of_rows; $row++){
            for($col = 0; $col < $number_of_columns; $col++){
                
                if($pixels[$curr_pixel] > 0){
                    $color = $white;
                }
                else{
                    $color = $black;
                }
                
                imagesetpixel($im, $col, $row, $color);
                $curr_pixel++;
            }
        }

        imagepng($im, $save_directory . DIRECTORY_SEPARATOR . "$curr_image.png");
        imagedestroy($im);
    
    } // for curr image
    
       $raw_bytes = NULL;
    unset($raw_bytes);
}


// / Functions
/////////////////////////////////////////////////////////////////////////////////

// Paths
// Packed Labels
$training_labels_file = __DIR__ . DIRECTORY_SEPARATOR . 'Training Data' . DIRECTORY_SEPARATOR . 'train-labels.idx1-ubyte';
$test_labels_file = __DIR__ . DIRECTORY_SEPARATOR . 'Training Data' . DIRECTORY_SEPARATOR . 't10k-labels.idx1-ubyte';

// Packed Images
$training_images_file = __DIR__ . DIRECTORY_SEPARATOR . 'Training Data' . DIRECTORY_SEPARATOR . 'train-images.idx3-ubyte';
$test_images_file = __DIR__ . DIRECTORY_SEPARATOR . 'Training Data' . DIRECTORY_SEPARATOR . 't10k-images.idx3-ubyte';

// Where to Unpack Data To
$train_directory = __DIR__ . DIRECTORY_SEPARATOR . 'Training Images' . DIRECTORY_SEPARATOR . 'train';
$test_directory = __DIR__ . DIRECTORY_SEPARATOR .  'Training Images' . DIRECTORY_SEPARATOR . 'test';


// Make sure the locations we're unpacking the images to exist
if (!mkdir($train_directory, 0777, true)) {
    die("Failed to create $train_directory");
}

if (!mkdir($test_directory, 0777, true)) {
    die("Failed to create $test_directory");
}



//////////////////////////
// Labels               //
//////////////////////////

// Extract labels from bytes to minst_train_labels.txt
ExtractLabelFromByteData($training_labels_file, $train_directory, 'minst_train_labels.txt');
ExtractLabelFromByteData($test_labels_file, $test_directory, 'minst_test_labels.txt');



//////////////////////////
// Images               //
//////////////////////////
    
// Extract images from bytes to 0.png, 1.png 2.png, ...
ExtractImagesFromByteData($training_images_file, $train_directory);
ExtractImagesFromByteData($test_images_file, $test_directory);


echo 'All Done!' . PHP_EOL;