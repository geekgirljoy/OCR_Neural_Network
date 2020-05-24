<?php 

/////////////////////////////////////////////////////////////////////////////////
// Functions


/////////////////////////////////////////////////////////
// Convolve()                                          //
/////////////////////////////////////////////////////////
// wrapper for the GD Lib imageconvolution() function 
function Convolve($input_image, $kernel){
    
    // How big is the input image?
    $width = imagesx($input_image);
    $height = imagesy($input_image);
    
    // Create an image resouce to convolve
    $convolution_image = imagecreatetruecolor($width,$height);      
    
    // Copy Input image to the convolution image
    imagecopy($convolution_image, $input_image, 0, 0, 0, 0, $width, $height);
    
    // Calculate the divisor
    $divisor = array_sum(array_map('array_sum', $kernel));
    
    // Convolve over the image
    imageconvolution($convolution_image, $kernel, $divisor, 0); 

    return $convolution_image; // Return the convolved image
}


/////////////////////////////////////////////////////////
// GenerateDatasetFromLabeledImages()                  //
/////////////////////////////////////////////////////////
// From MNIST images and labels, create a FANN .data training file
define('CONVOLUTION_LAYER', 0);
define('POOLING_LAYER', 1);
define('FLATTENING_LAYER', 2);
function GenerateDatasetFromLabeledImages($directory, $labels_file, $save_data_directory, $save_data_file, $layers, $kernels, $pool_type = MAX_POOL, $pooling_size){
    
    $error_log = array(); // Error Log

    // Load the labels file
    $data = file_get_contents($directory . DIRECTORY_SEPARATOR . $labels_file);
    $data = explode(PHP_EOL, $data);
    
    $labels = array();
    // Split image file name and label
    foreach($data as $label){
        $label = explode(' ', $label);
        $labels[] = array('image'=> $label[0], 'label'=>$label[1]);
    }
    $data = NULL;
    unset($data);
    
    // Open a temporary file to store our data
    $temp_file = @fopen($save_data_directory . DIRECTORY_SEPARATOR . 'temp.data', "w+");
    
    $number_of_sets = 0;
    $number_of_inputs = NULL;
    
    // Loop through the set of images and create dataset
    foreach($labels as $set){

        $image = $set['image'];
        $label = $set['label'];

        // If this image is not in the directory
        if(!file_exists($directory . DIRECTORY_SEPARATOR . $image)){
            $error_log[] = "Missing Image: $image is missing in $directory"; // Log the error
        }
        else{ // Image is available
            $number_of_sets++;

            // Load image
            if(pathinfo($directory . DIRECTORY_SEPARATOR . $image, PATHINFO_EXTENSION) == 'png'){
                $img = imagecreatefrompng($directory . DIRECTORY_SEPARATOR . $image);
            }
            elseif(pathinfo($directory . DIRECTORY_SEPARATOR . $image, PATHINFO_EXTENSION) == 'jpg'
                   || pathinfo($directory . DIRECTORY_SEPARATOR . $image, PATHINFO_EXTENSION) == 'jpeg'){
                $img = imagecreatefromjpeg($directory . DIRECTORY_SEPARATOR . $image);
            }

            $image_width = imagesx($img);  // 28px
            $image_height = imagesy($img); // 28px

            $flattened_images = array();

            foreach($kernels as $key=>$kernel){

                 // Create an image resouce to convolve
                $temp = imagecreatetruecolor(imagesx($img),imagesy($img));      
                imagecopy($temp, $img, 0, 0, 0, 0, $image_width, $image_height);

                foreach($layers as $layer){
                    if($layer == CONVOLUTION_LAYER){
                        $temp = Convolve($temp, $kernel);
                    }
                    elseif($layer == POOLING_LAYER){
                        $temp = PoolImage($temp, $pooling_size, $pool_type);
                    }
                    elseif($layer == FLATTENING_LAYER){
                        $flattened_images[$key] = FlattenImage($temp, array(RGB_CHANNEL_AVERAGE));
                    }
                }

                // Destroy temp image resource
                imagedestroy($temp);
            }

            // Flatten Images into merged array
            while(is_array($flattened_images[array_key_first($flattened_images)])){
                $flattened_images = call_user_func_array('array_merge', $flattened_images);
            }

            // Destroy input image resource
            imagedestroy($img);

            // Prepare flattened pixel data 
            foreach($flattened_images as &$pixel){
                // MNIST data pixel data should be:
                // black 0 
                // white 255
                // Converted to a float:
                // black -1.00
                // white 1.00
                // so we can easily say if 
                // it's less than 0.9 it zero
                // and be right
                if(ColorToFloat($pixel) > 0){
                    $pixel = 1; // White pixel
                }
                else{ // Not White pixel
                    $pixel = -1;
                }
            }

            if($number_of_inputs == NULL){
                $number_of_inputs = count($flattened_images);
                
                // Only echo the number of inputs once
                static $reported_number_of_inputs = false;
                if($reported_number_of_inputs == false){
                    $reported_number_of_inputs = true;
                    echo "ANN Number of Inputs Required: $number_of_inputs" . PHP_EOL; 
                }
            }

            // Merge flattened_images into a space delimited string
            $flattened_images = implode(' ', $flattened_images);

            // Convert number label to outputs string
            // Example Outputs:
            // 0 = 1000000000
            // 1 = 0100000000
            // 2 = 0010000000
            // 3 = 0001000000
            // 4 = 0000100000
            // 5 = 0000010000
            // 6 = 0000001000
            // 7 = 0000000100
            // 8 = 0000000010
            // 9 = 0000000001
            // Keep this order of operations to maintain the correct label to output generation
            // If label is 0 this adds nothing and if >= 1 adds the correct number of preceding zeros


            $outputs = str_repeat('0', $label);
            $outputs .= str_repeat('1', 1); // Add 1 at the correct location
            $outputs .= str_repeat('0', 9 - $label); // Add 9 - the number of the label to pad to the end
            $outputs = str_split($outputs); // string to array
            $outputs = str_replace("0","-1", implode(' ', $outputs)); // convert the array back to a string
                                                                      // and add spaces between the
                                                                      // values and convert 0's to -1's

            // Example:
            // $outputs = '1 0 0 0 0 0 0 0 0 0';
            // Write Data
            fwrite($temp_file, PHP_EOL . $flattened_images  . PHP_EOL . $outputs);

        } // For Each Image

    } // For all the labels sets

    // Move back to the beginning of the temp file
    fseek($temp_file, 0);

    /////////////////////////////////////////////////////////////////////
    // Add FANN Header to data and migrate temp data into data file.   //
    /////////////////////////////////////////////////////////////////////

    // Create Training Data File
    $data_file = @fopen($save_data_directory . DIRECTORY_SEPARATOR . $save_data_file, 'w'); // r+

    // Write FANN Header
    fwrite($data_file, "$number_of_sets $number_of_inputs 10");

    // Transfer Data from temp file to the data file
    while (($buffer = fgets($temp_file, 4096)) !== false) {
        fwrite($data_file, $buffer);
    }

    if(!feof($temp_file)){ // Something went wrong and the transfer was incomplete
        $error_log[] = 'File Write Error: Unexpected fgets() fail when writing data from temp file.';
    }
	
    // Close files
    fclose($temp_file);
    fclose($data_file);
    
    // Delete temp file
    if(!unlink($save_data_directory . DIRECTORY_SEPARATOR . 'temp.data')){
        $error_log[] = 'Unlink Error: Unable to remove temp file.';
    }
    
    // If there are any errors
    if(count($error_log) > 0){
        // Create error log file
        $error_log_file = @fopen($save_data_directory . DIRECTORY_SEPARATOR . "error.log.$save_data_file.txt", 'w');
        fwrite($error_log_file, implode(PHP_EOL, $error_log));
        fclose($error_log_file);
    }

} // / GenerateDatasetFromLabeledImages()
        

/////////////////////////////////////////////////////////
// PoolImage()                                         //
/////////////////////////////////////////////////////////
/*
Examples:
// 28 x 28 input image
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000


// 28 x 28 image pooled with a 2 grid results = 14x14 px output pooled image

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00

00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00



// 28 x 28 image pooled with a 7 grid results = 4x4 px output image
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000

0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000

0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000

0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000
0000000 0000000 0000000 0000000



// 28 x 28 image pooled with a 14 grid results = 2x2 px output image
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000

00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
00000000000000 00000000000000
*/
define('MAX_POOL', 0);
define('MIN_POOL', 1);
define('AVG_POOL', 2);
function PoolImage($img, $grid_size, $pool_type = MAX_POOL){
   
   
    ///////////////////////////////////
    // Check For Errors              //
    ///////////////////////////////////
       
    // Check if $img is a valid image resource
	if(get_resource_type($img) != 'gd'){
		die('PoolImage(image, grid_size, pool_type) - image must be a valid gd image resource.' . PHP_EOL);
	}
	
	// Get the size info from the input image
    $width = imagesx($img);
    $height = imagesy($img);
	
	
	// Check for Grid Size Errors
	if(!is_numeric($grid_size)){ // Not an a number
		die('PoolImage(image, grid_size, pool_type) - grid_size must be an int value.' . PHP_EOL);
	}
	
	if(is_float($grid_size)){ // Number is a float
		$grid_size = round($grid_size);
		echo "PoolImage(image, grid_size, pool_type) - grid_size rounded to $grid_size." . PHP_EOL;
	}
	
	if($grid_size <= 1){ // Less than or equals to 1
		$grid_size = 2;
		echo "PoolImage(image, grid_size, pool_type) - $grid_size is too small, increased to 2." . PHP_EOL;
	}
	
	if(is_float($width / $grid_size) || is_float($height / $grid_size)){ // grid size results in incomplete matrix
		echo "PoolImage(image, grid_size, pool_type) - grid_size $grid_size results in incomplete matrix, attempting to automatically resize... ";
		
		// Attempt to automatically resize grid
		while( (is_float($width / $grid_size) || is_float($height / $grid_size)) // while grid size won't work
			   && ($grid_size <= ($width / 2) && ($grid_size <= ($height / 2)))  // and while less than or equal to 1/2 image width and height 
			 ){
			$grid_size++; // make the grid_size larger
		}
		
		// Check if the grid resize was not successful
		if(is_float($width / $grid_size) || is_float($height / $grid_size)){
			die('Unsuccessful!' . PHP_EOL);
		}
		else{
			echo "Successful! - grid_size resized to $grid_size" . PHP_EOL;
		}
	}
	
	// Check for Pool Type Errors
	$valid_pool_types = array(MAX_POOL, MIN_POOL, AVG_POOL);
	if(!in_array($pool_type, $valid_pool_types)){
		die('PoolImage(image, grid_size, pool_type) - Invalid type selected, use: MAX_POOL, MIN_POOL or AVG_POOL' . PHP_EOL);
	}

    ///////////////////////////////////
    // No Errors Detected            //
	///////////////////////////////////

    // Determine the size of the pool image.
    $pool_img_width = $width / $grid_size;
    $pool_img_height = $height / $grid_size;   

    // Allocate resource in memory for the image
    $pool_img = imagecreatetruecolor($pool_img_width, $pool_img_height) or die('Cannot Initialize new GD image stream');
   
    // Copy Input image to the convolution image
    imagecopy($pool_img, $img, 0, 0, 0, 0, $width, $height);
	
   
    // Keep track of where in the image we are
    $curr_row = 0;
    $curr_col = 0;

    // Pooling - Loop through the entire image in grid matrix chunks
    for($row = 0; $row < $width; $row += $grid_size){
        for($col = 0; $col < $height; $col+= $grid_size){
            
			$pool_matrix = array();	
            
			// Max & Min pooling grid matrix chunk
			if($pool_type == MAX_POOL || $pool_type == MIN_POOL){
				for($i = $row; $i < $row+$grid_size; $i++){
					for($k = $col; $k < $col+$grid_size; $k++){

						$p = imagecolorat($img, $i, $k);
					
						$colors['red'] = ($p >> 16) & 0xFF;
						$colors['green'] = ($p >> 8) & 0xFF;
						$colors['blue'] = $p & 0xFF;

						// Key is color "amplitude" R+G+B.
						$key = $colors['red']+$colors['green']+$colors['blue'];

			
						// We are only concerned with finding the "brightest" (max) or "darkest" (min) pixel 
						// so identical pixels or equal strength/amplitude are treated as duplicates and ignored
						// so if a combined RGB 100+0+0 exists in the array already and 0+100+0 is encountered
						// it is ignored because we already have a pixel at that amplitude;
                        // We could add a channel preference option but at this time I'm not going to.						
						if(!in_array($key, $pool_matrix)){
							// pixel key = R+G+B (i.e. the strength of all the color channels combined)
							// the value is a the color
							$pool_matrix[$key] = $colors;
						}
					}
				}

				if($pool_type == MAX_POOL){
					// Find the brightest pixel
					$pooled_pixel = max(array_keys($pool_matrix));
				}
				else{ // MIN_POOL
				    // Find the darkest pixel
					$pooled_pixel = min(array_keys($pool_matrix));
				}
			}
			else if($pool_type == AVG_POOL){ // Average pooling

			    $colors = array('red'=>0, 'green'=>0, 'blue'=>0);

				for($i = $row; $i <= $row+$grid_size; $i++){
					for($k = $col; $k <= $col+$grid_size; $k++){
						$p = imagecolorat($img, $i, $k);
						$colors_temp = imagecolorsforindex($img, $p);
								
						// Add this pixels colors to the channel totals
						$colors['red'] += $colors_temp['red'];
						$colors['green'] += $colors_temp['green'];
						$colors['blue'] += $colors_temp['blue'];
					}
				}

				// Find the average of all the pixels in this chunk
				// Divide each color channel by the number of pixels sampled
                // to get the average color channel strength of all pixels
				$number_of_sampled_pixels = $grid_size * $grid_size;
				$colors['r'] = $colors['r'] / $number_of_sampled_pixels;
				$colors['g'] = $colors['g'] / $number_of_sampled_pixels;
				$colors['b'] = $colors['b'] / $number_of_sampled_pixels;

				$pooled_pixel = 0;

				// Allocate a new color and store it 
				$pool_matrix[$pooled_pixel] = $colors;
			}
			else{
				die('Unknown Pool type selected. Use: MAX_POOL, MIN_POOL, AVG_POOL' . PHP_EOL);
			}

			$r = $pool_matrix[$pooled_pixel]['red'];
			$g = $pool_matrix[$pooled_pixel]['green'];
			$b = $pool_matrix[$pooled_pixel]['blue'];

			// Set Pixel
			$color = imagecolorallocate($pool_img, $r, $g, $b);

            // Paint pooled pixel
            imagesetpixel($pool_img, $curr_row, $curr_col, $color);
            $curr_col++;
        }
        $curr_col = 0;
        $curr_row++;
    }

    return $pool_img;
} // / PoolImage()


/////////////////////////////////////////////////////////
// FlattenImage()                                      //
/////////////////////////////////////////////////////////
// Take a 2D image and return a 1D array 
// of arrays containing the desired color channels as int’s
// Note: Channels are returned in the order requested
//
// i.e. pixel: r = 5, g = 22, b = 7
//
// Example: FlattenImage($img, array(RED_CHANNEL, BLUE_CHANNEL))
// Would Return: array(0=>array(5, 7))
// 
// Example: FlattenImage($img, array(BLUE_CHANNEL, GREEN_CHANNEL))
// Would Return: array(0=>array(7, 22))
// 
// Example: FlattenImage($img, array(RGB_CHANNEL_AVERAGE))
// Would Return: array(0=>array(11))
// round(5+22+7 / 3) = 11
// Example: FlattenImage($img, array(RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL, RGB_CHANNEL_AVERAGE))
// Would Return: array(0=>array(5, 22, 7, 11))
//
// Default returned is array(0=>array(R, G, B), 1=>array(R, G, B), ...)
//
define("RED_CHANNEL", 0);   // Only Red Channel
define("GREEN_CHANNEL", 1); // Only Green Channel
define("BLUE_CHANNEL", 2);  // Only Blue Channel
define("RGB_CHANNEL_AVERAGE", 4);   // R+G+B/3
function FlattenImage($img, $color_channels = array(RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL)){

	///////////////////////////////////
    // Check For Errors              //
    ///////////////////////////////////


    // Check if $img is a valid image resource
	if(get_resource_type($img) != 'gd'){
		die('Flatten(image, color_channels) - image must be a valid gd image resource.' . PHP_EOL);
	}

	foreach($color_channels as $key=>$channel){
		if(!in_array($channel, array(RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL, RGB_CHANNEL_AVERAGE))){
			die('Flatten(image, color_channels) - color_channels must be a valid channel type, use: ' 
                . 'RED_CHANNEL, GREEN_CHANNEL, BLUE_CHANNEL, RGB_CHANNEL_AVERAGE' . PHP_EOL);
		}
	}

	///////////////////////////////////
    // No Errors Detected            //
	///////////////////////////////////

	///////////////////////////////////
    // Flatten The Image             //
	///////////////////////////////////

    // Get the size info from the input image
    $width = imagesx($img);
    $height = imagesy($img);

    // the flattened pixel data is stored here
    $pixels = array();

    // Loop through pixels
    for($row = 0; $row < $width; $row++){
        for($col = 0; $col < $height; $col++){

            // Get pixel color channels 
            $p = imagecolorat($img, $row, $col);

			$colors['red'] = ($p >> 16) & 0xFF;
			$colors['green'] = ($p >> 8) & 0xFF;
			$colors['blue'] = $p & 0xFF;

			$p = array();
			foreach($color_channels as $key=>$channel){

				// Extract desired channels
				if($channel == RED_CHANNEL){
					$p[] = $colors['red'];
				}
				elseif($channel == GREEN_CHANNEL){
					$p[] = $colors['green'];
				}
				elseif($channel == BLUE_CHANNEL){
					$p[] = $colors['blue'];
				}
				elseif($channel == RGB_CHANNEL_AVERAGE){
					$p[] = ($colors['red']+$colors['green']+$colors['blue']) / 3;
				}
			}
			$pixels[] = $p;
        }
    }

    return $pixels;
} // / FlattenImage()


/////////////////////////////////////////////////////////
// ColorToFloat()                                      //
/////////////////////////////////////////////////////////
// Input: 0 - 255 
// Output: 0.00 - 1.00
function ColorToFloat($value){

	// Only convert valid colors
	if($value < 0){
		$value = 0;
	}
	// Only convert valid colors
	if($value > 255){
		$value = 255;
	}

    $max = 255;
    $increment = $max / 100;
    $value = ($value / $increment) / 100;

	// If the value isn't a float
	if(!is_float($value)){ // value is 0 or 1
		$value .= '.00';   // append dot zero zero
	}
    return $value;
}



/////////////////////////////////////////////////////////
// FlattenANNLayers()                                  //
/////////////////////////////////////////////////////////
// Input Array:
/*
array(3) {
  [0]=>
  int(50)
  [1]=>
  array(3) {
    [0]=>
    int(200)
    [1]=>
    int(300)
    [2]=>
    int(10)
  }
  [2]=>
  int(10)
}
*/
// Output Array:
/*
array(5) {
  [0]=>
  int(50)
  [1]=>
  int(200)
  [2]=>
  int(300)
  [3]=>
  int(10)
  [4]=>
  int(10)
}
*/

function FlattenANNLayers($layers){
    for($i = 0; $i < count($layers); $i++){
        if(is_array($layers[1])){
            $layers[0] = array_merge(array($layers[0]), $layers[1]);
        }
        else{
            $layers[0][] = $layers[1];
        }
        unset($layers[1]);
        $layers = array_values($layers);
    }
    return array_values($layers[0]);
}


// / Functions
/////////////////////////////////////////////////////////////////////////////////
