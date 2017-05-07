<?php
/*
    GenerateTrainingData(
                            (NULL)
                        );
    
    Description: 
        
    This function manages the image creation, call it to create a new image training set.
    
    The images will be 10px wide and 16px tall.
    
    [Example: Capital A Training Image]
    
      Pixels     Encoded
    ██████████  0000000000
    ██████████  0000000000
    ██████████  0000000000
    ████  ████  0000110000
    ███    ███  0001111000
    ██  ██  ██  0011001100
    █  ████  █  0110000110
    █  ████  █  0110000110
    █  ████  █  0110000110
    █        █  0111111110
    █  ████  █  0110000110
    █  ████  █  0110000110
    █  ████  █  0110000110
    ██████████  0000000000
    ██████████  0000000000
    ██████████  0000000000

    
    [Example: Capital A Prepared for ANN ]  
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 1 1 0 0 0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 1 0 0 1 1 1 1 1 1 1 1 0 0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0.32
    
    This algorithm works by iterating over an image pixel by pixel and evaluating the RBG color values of a pixel in an image and assigning a single numerical value to represent the color of the pixel.
    
    White pixels will be encoded as 1 and black pixels will be 0.
    
    Counts begin with 0.
    
    We proceed from the top left corner of the image which for descriptive purposes could be referred to as $row 0, and $col 0 (0,0) and the bottom most right pixel would be the last processed and would be (15,9). 
    
    All columns on a given row will be encoded prior to proceeding to the next row.
  
    As it is currently written the images should be in a sub-folder named 'images' and the image set should 
    be named from 0.png to the last image in the set e.g. 93.png or other if you changed the code that generates
    the training images.
    
    
    References:       
       file_exists() - http://php.net/manual/en/function.file-exists.php
       die() - http://php.net/manual/en/function.die.php
       fopen() - http://php.net/manual/en/function.fopen.php
       fgets() - http://php.net/manual/en/function.fgets.php
       PHP_EOL - http://php.net/manual/en/reserved.constants.php
       str_replace() - http://php.net/manual/en/function.str-replace.php
       array_push() - http://php.net/manual/en/function.array-push.php
       explode() - http://php.net/manual/en/function.explode.php
       feof() - http://php.net/manual/en/function.feof.php
       fclose() - http://php.net/manual/en/function.fclose.php
       getimagesize() - http://php.net/manual/en/function.getimagesize.php
       count() - http://php.net/manual/en/function.count.php
       fwrite() - http://php.net/manual/en/function.fwrite.php
       imagecreatefrompng() - http://php.net/manual/en/function.imagecreatefrompng.php
       imagecolorat() - http://php.net/manual/en/function.imagecolorat.php
       imagecolorsforindex() - http://php.net/manual/en/function.imagecolorat.php
       fwrite() - http://php.net/manual/en/function.fwrite.php
       imagedestroy() - http://php.net/manual/en/function.imagedestroy.php
       fclose() - http://php.net/manual/en/function.fclose.php
       
*/
function GenerateTrainingData() {
    
    /* 
        We will use the $image_array variable to store all the image data as an array.
    
        $image_array[$i][0] = File name / ASCII number
        $image_array[$i][1] = ASCII symbol
        $image_array[$i][2] = Desired output value from the ANN as a floating point number between -1 & 1
        $image_array[$i][3] = Encoded pixel data
    */
    $image_array = array();
    
    /* If no training images don't proceed. Determined by checking for images/0.png */
    if (!file_exists('images/0.png')) {
        /* No Image so do not proceed with the encoding. Output a hyperlink to generate_training_images.php */
        die('No training images run <a href="generate_training_images.php">generate_training_images.php</a> first.');
    }
    
    /* Create an empty file resource that points to ocr.data (where the training data will be stored) */
    $trainingfile = fopen("ocr.data", "w") or die("Unable to open: " . $trainingfile . '. Ending program.');
    
    /* Create a file resource that points to generate_images.log */
    $logfile = fopen("images/generate_images.log", "r") or die("Unable to open: " . $logfile . '. Ending program.');
    
    
    /*
        Use fgets() to open then buffer generate_images.log line by line.
        
        Use str_replace() on the buffered data to remove line ending terminators.
        
        Use explode() to split the remaining buffered data using spaces ' ' as
        delimiter into $image_array
                        
        Example generate_images.log excerpt:
        
        0 !
        1 "
        2 #
        3 $
        4 %
        5 &
        6 '
        7 (
        .......
    */    
    while (($buffer = fgets($logfile, 4096)) !== false) {
        $buffer = str_replace(PHP_EOL, '', $buffer);
        array_push($image_array,  explode(' ', $buffer));
    }
    if (!feof($logfile)) {
        echo "Error: unexpected fgets() fail while reading logfile.\'n";
    }
    
    /* Close the logfile */
    fclose($logfile);
    
    
    /* 
       Use getimagesize() to obtain the width and height of the training images.
       
       We could just hand code these but I wanted to demonstrate how you could 
       programmatically determine the image dimensions.
       
       imgsize[0] = width (10)
       imgsize[1] = height (16)
    */
    $imgsize = getimagesize('images/0.png');
    
    
    /*       
       Next we programmatically determine the number of inputs for the ANN by computing
       the area of the image in pixels. The area of a rectangle is Width * Height, therefore:
       
       10 * 16 = 160 (pixels / inputs) 
    */
    
    $num_of_inputs = $imgsize[0] * $imgsize[1]; 
    
    /* Determine how many images there are */
    $num_of_images = count($image_array);
    
    /* 
        Start writing to the training data file.        
        Write: $num_of_images $num_of_inputs 1
    */
    fwrite($trainingfile, "$num_of_images $num_of_inputs 1" . PHP_EOL);
    
    
    /* Process each image */
    for($i = 0; $i < $num_of_images; $i++){
        $curr_image = "images/$i.png";
        
        /* Load the training image into memory */
        $im = imagecreatefrompng($curr_image);
        
        /* 
            Determine the desired output value for this training image.
            
            Use array_push() to add the desired output to $image_array[$i][2]
        */
        if($i > 0){
            $output_value = 0.01 * $i;
        }else{
            $output_value = 0.00;
        }
        array_push($image_array[$i], $output_value); 
        
        
        
        /* 
            Step through the image and look at each pixel using imagecolorat().
            
            Use imagecolorsforindex() to split $rgb resource to separate $colors array.
            
            Assign the pixel a single value based on its RGB color.
            
            Concatenate all the pixel values into the $pixel_values variable.
        */
        $pixel_values = "";
        for($row = 0; $row < $imgsize[1]; $row++){
            for($col = 0; $col < $imgsize[0]; $col++){
                $rgb = imagecolorat($im, $col, $row);
                $colors = imagecolorsforindex($im, $rgb);
                
                if($colors['red'] >= 225 && $colors['green'] >= 225 && $colors['blue'] >= 225){
                    $pixel_values .= 1 . ' ';
                } else{
                    $pixel_values .= 0 . ' ';
                }
            }
        }
        /*
            Once every pixel has been scanned and encoded for use as inputs
            use array_push() to add the pixel value inputs to $image_array[$i][3]
        */
        array_push($image_array[$i], $pixel_values);
        
        /* Echo links and values for the image */
        echo "<a href='images/" . $image_array[$i][0] . ".png' target='_blank'>" . $image_array[$i][0] . "</a>.png encoded as " . $image_array[$i][2] . "<br>"  . PHP_EOL;
        echo "Pixel Data: " . $image_array[$i][3] . "<br>"  . PHP_EOL;
    
        /* Write the inputs and desired outputs to the training data file */    
        fwrite($trainingfile, $image_array[$i][3] . PHP_EOL . $image_array[$i][2] . PHP_EOL);
        
        /* Free up memory associated with the training image by destroying the resource. */ 
        imagedestroy($im);
    }
    /* All done! Close the training data file. */
    fclose($trainingfile);
}

/* Generate training data from training images. */
GenerateTrainingData();

/* In case the user wishes to review the training data file link to it. */
echo 'Training data: <a href="ocr.data">ocr.data</a><br>' . PHP_EOL;

/* Announce completion and link to next step. */
echo 'All Done! Now run <a href="train_ocr.php">Train OCR</a><br>' . PHP_EOL;



?>
