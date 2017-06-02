# OCR_Neural_Network
![OCR Logo](https://github.com/geekgirljoy/OCR_Neural_Network/blob/ca5dbcd3198dd36d2bbbbbc3731e450cac7ef7a4/Project%20Images/OCR.png)

OCR is a practical example of Optical Character Recognition using [FANN](https://github.com/bukka/php-fann). While this example is limited and does make mistakes, the concepts illustrated by OCR can be applied to a more robust stacked network that uses feature extraction and convolution layers to recognize text of any font in any size image.

**At the end of this series of tutorials you will be able to build Neural Networks using PHP that can read characters from images! I will be giving you actual working code!**

As mentioned this will be a series of posts so that I don’t overwhelm you guys with too much information all at once and so I don’t have to sit here and type [ad infinitum](https://en.wikipedia.org/wiki/Ad_infinitum) (infinite recursion).

😛

[OCR (Optical Character Recognition)](https://en.wikipedia.org/wiki/Optical_character_recognition) isn’t exactly a new subject but surprisingly its something that few computer scientists have actual experience building! Further, any examples you see are descriptions at best that frequently “devolve” into a math lesson that ultimately glosses over practical application and important implementation details!

Don’t get me wrong, I love math but that isn’t required to start learning. The FANN Library will act as an abstraction layer so we can focus on our data and objectives and not complex differential equations.

Additionally all the code will be thoroughly documented and intentionally simplified and referenced so that even a student with minimal experience can benefit. I happen to believe that Neural Networks are complex enough already, and the more people who know how to build and deploy these systems the faster we can find solutions to the most horrible problems we face as a species (currently incurable diseases, famine, wars over resources, the global energy crisis) need I go on?

So, I am going to provide you with the tools and basic knowledge of how to build POWERFUL artificial intelligence systems and deploy them to cloud servers, not to shabby eh? 😉

Not so you can go get rich building games and businesses (which you could easily do and that’s fine) but it is my very real hope that at lease some of you can apply these tools to help make the world a better place, start in your own community today!

In a very surreal way I am reminded of a lyric from the song [“I’d Love to Change the World” by Ten Years After](https://www.youtube.com/watch?v=ZyaFeDlJJAk)

*“I’d love to change the world, But I don’t know what to do, So I’ll leave it up to you-ooo-ooo”*

Its rather haunting in how true that actually is, isn’t it?

If you do happen to start a business using these tools and techniques or you simply appreciate the content that I create please support me on [Patreon](https://www.patreon.com/user?u=3969727) and please share this project with your followers, friends and coworkers on social media.

Now to begin you will need an environment to build your [ANN (Artificial Neural Network)](https://en.wikipedia.org/wiki/Artificial_neural_network), rather than reproduce the steps here you can follow this tutorial I wrote to get setup to work: [Getting started with Neural Networks using the FANN library, PHP and C9.io](https://geekgirljoy.wordpress.com/2016/07/12/getting-started-with-neural-networks-using-the-fann-library-php-and-c9-io/)

Go follow the setup process and come back, I’ll wait. 😉

So, in order for a Neural Network to be useful it needs a “problem space” (basically the thing we want it to learn or do) and a “training set” (examples of data similar to the data the ANN will encounter in the “problem space” but which already has a known value or solution). In this case, because we are teaching the ANN to read characters from images, we need a set of images with alpha-numeric characters in them.

Further, before getting started we have to make sure that we don’t violate the licensing of any training set we use so to keep things simple for this tutorial I will show you how to generate your own basic training data programmatically.

Because the training set will be generated programmatically rather than by hand we are excluding hand written characters from this example ANN however with the use of additional training sets and [convolution layers](https://en.wikipedia.org/wiki/Convolutional_neural_network) you could add that functionality to this neural network, however again, I want this example to be understandable by everyone.

So with that being said, lets begin by generating the training set of images!

We can use the [PHP GD library](http://php.net/manual/en/book.image.php)  which is a library used to manipulate images.

The GD Lib is almost always installed (compiled into php) by your host for you already and you probably have used it in the past (even if you didn’t realize it) for your other projects so I wont cover it in too much detail, suffice it to say its pretty east to get access to.

The images in our simple training set will be 10 pixels wide and 16 pixels high.

(10*16 = 160 pixels per image)

And in this example I will use only black and white for simplicity, white will be 1 and black will be 0. It’s completely arbitrary and you could reverse the colors if you wanted but white text on black seems to be high contrast so I decided to use that.

It’s also worth noting the importance of understanding that fundamentally there is no reason why you can’t use floating point values and represent the entire color spectrum or just gray scale.

Switching to a float would allow you to encode more information as a gradient however this will increase complexity of your ANN and you may require more hidden neurons, layers or both in addition to increased training epochs. In reality you would likely be MUCH better off adding [convolutions](https://en.wikipedia.org/wiki/Convolutional_neural_network) and building a more robust ANN.

This ANN does not use convolution layers and will make a few mistakes from time to time, my point is that this is a simplified, stripped down, easy to understand example however with a little work you could build this into a very robust OCR ANN.

So now lets look at a sample training image as well as dive into the code to generate them!
![alt text](https://github.com/geekgirljoy/OCR_Neural_Network/blob/ca5dbcd3198dd36d2bbbbbc3731e450cac7ef7a4/Project%20Images/TrainingImageInfoGraphic.png)

## [generate_training_images.php](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Basic/generate_training_images.php)

```php
<?php

/*
    NewImage( 
	          (int)$char, 
	          (mixed)$curr_image, 
			  (int)$x , 
			  (int)$y, 
			  (file resource)$logfile
			);
	
	Description: 
	
	This function actually creates the images. 
	
	It is written with an iterative process in mind where more than a single training image (batch creation) would be generated however a single image can be generated if you prefer.
	
	I have created the GenerateTrainingImages() function that will call this function for you so you *could*
	ignore this function but its the one that does the real work. 
		
	$char uses the chr() function to convert the number passed to an ASCII character.
	
	$curr_image is a mixed type variable used to name the image that is being generated. In this
	case (i.e how I used it in GenerateTrainingImages()) I have used it as a number so it's easy to 
	iterate over the image files using a simple for loop to programmatically create the images we
    need however you could create an array of strings or chars and use those instead however that is
	not quite as simple as the implementation shown here.
	
	$x & $y are direct pass-through variables for the x & y coordinate placement variables as defined
	in the imagestring() function documentation: http://php.net/manual/en/function.imagestring.php
    
	The main reason why I made them "pass-through" rather than hard coding them in the NewImage()
    function is simply that you may want to vary or stagger the placement of the character within the images.
	
	$logfile is a file resource variable that points to the file that will log results of generating
	the training images. You may want to use this file in later steps or for reference. Please note that 
	NewImage() does not open its own access to the file so you will need to open the file resource yourself
	prior to calling NewImage().
	
	
	Example NewImage() Usage:
	
	// Create $logfile resource
	$logfile = fopen("images/generate_images.log", "w") or die("Error: Unable to open: " . $logfile . '. Ending program.');
	
	// This will create a b/w image of an exclamation mark(!) named 0.png
	NewImage(33, 0, 1, 0, $logfile);
	
	// Close log file 
    fclose($logfile);
	
	
	References:	   
	   chr() - http://php.net/manual/en/function.chr.php
	   imagecreate() - http://us.php.net/manual/en/function.imagecreate.php
	   imagecolorallocate() - http://us.php.net/manual/en/function.imagecolorallocate.php
	   imagestring() - http://us.php.net/manual/en/function.imagestring.php
	   imagepng() - http://us.php.net/manual/en/function.imagepng.php
	   imagedestroy() - http://us.php.net/manual/en/function.imagedestroy.php
	   fwrite() - http://us.php.net/manual/en/function.fwrite.php
*/
function NewImage($char, $curr_image, $x , $y, $logfile){
    /* Size the images */
    $width = 10; // px
    $height= 16; // px
	
	/* Set the filename */
    $file_name = $curr_image . '.png';


    /* Create the image resource 
	
	   Note: The @ operator is used to suppress any errors generated by php expressions. 
	         http://us.php.net/manual/en/language.operators.errorcontrol.php
			 
	         I use it to suppress any errors from @imagecreate() and instead cast my own
			 error message by adding an "or die()" to the @imagecreate() statement.
	*/
    $image = @imagecreate($width, $height) or die("Error: Unable to Initialize Image Stream");
	
	
	/* Add colors to the image resource	
	   
	   Note: colors are defined by the RBG color model
	   https://en.wikipedia.org/wiki/RGB_color_model

	   RGB Black: (0, 0, 0)
	   RBG White: (255, 255, 255)	   
	*/
    $background_color = imagecolorallocate($image, 0, 0, 0);
    $text_color = imagecolorallocate($image, 255, 255, 255);
	
	/* Add $char to the image resource */
    imagestring($image, 5, $x, $y,  chr($char), $text_color);
	
    /* Draw the image buffer stream to the file */
    imagepng($image, './images/' . $file_name);
	
	/* Free the memory associated with the $image resource by using imagedestroy() */
    imagedestroy($image); 

    /* write to log file */ 
    fwrite($logfile, $curr_image . ' ' . chr($char). PHP_EOL);
	
	/* echo results and link to file for review */
	echo "<a href='images/$curr_image.png' target='_blank'>" . $curr_image . ".png</a> - " . chr($char) . " ...complete.<br>" . PHP_EOL;
}



/*
	GenerateTrainingImages(
	                        (NULL)
	                      );
	
	Description:
	
	This function manages the creation of the training images. Call this function to create a new training set. 

	
	Example GenerateTrainingImages() Usage:
	
	GenerateTrainingImages();
	
	
	References:
	file_exists() - http://us.php.net/manual/en/function.file-exists.php
	mkdir() - http://us.php.net/manual/en/function.mkdir.php
	fopen() - http://us.php.net/manual/en/function.fopen.php
	fclose() - http://us.php.net/manual/en/function.fclose.php
	
*/
function GenerateTrainingImages() { 

    /* Check if the "images" folder was already created */
    if (file_exists('images')) {
        echo "The images folder already exists, no changes to the images folder were made!<br>" . PHP_EOL;
    }
    else{ /* There is no "images" folder */
        echo "The images folder does not exists, creating one... ";
		
		/* try to create one with the correct folder permissions */
        if (!mkdir("images", 0755, true)) {
            die('fail! Ending program.'); /* Failed to create the folder */
        }else{
            echo 'success!<br>' . PHP_EOL; /* Successfully created the folder */
        }
    }

    /* Create log file resource to log the results of generating the training images */
    $logfile = fopen("images/generate_images.log", "w") or die("Unable to open: " . $logfile . '. Ending program.');

    /* Current number of generated images */    
    $curr_image = 0;

	
    /* 
        Training images set is defined in ASCII 
		https://en.wikipedia.org/wiki/ASCII
		
        start: ascii dec 33 (!) 
        stop:  ascii dec 126 (~) 
    */
	
	$start = 33;
	$stop = 126;
	$total = $stop - $start + 1; /* Add 1 because the count starts at 0 */
	echo "Starting batch creation of " .  $total . " images.<br><br>" . PHP_EOL;
	
    for($i = $start; $i <= $stop; $i++) {
        NewImage($i, $curr_image, 1, 0, $logfile);
        $curr_image++;
    }
	echo "<br>Batch complete.<br>" . PHP_EOL;
	echo "Log: <a href='images/generate_images.log' target='_blank'>generate_images.log</a><br>" . PHP_EOL;
    
	
    /* Close log file */
    fclose($logfile);
}

/* Kick the tires and light the fires! */
GenerateTrainingImages();

/* Announce completion and link to next step */
echo 'All Done! Now run <a href="generate_training_data.php">generate_training_data.php</a><br>' . PHP_EOL;

```


It looks like a lot of code, I know! If you are feeling overwhelmed delete my comments and you will see that the actual code is quite short and very simple.
If you run this code the following things will happen:

    1. A sub-folder called ‘images’ will be created for you where you ran generate_training_images.php.
    2. 94 training images will be generated for you and saved in the images sub-folder.
    3. A log file named ‘generate_images.log’ will be created.

At this point we are ready to use these images to create the training set that the Neural Network will use. 

Here is our newly minted [set of training images](https://github.com/geekgirljoy/OCR_Neural_Network/tree/master/Basic/images).

Next we will convert the training images into training data that the OCR ANN can use to learn the letters, numbers and symbols in our training images.


&nbsp;
## [generate_training_data.php](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Basic/generate_training_data.php)

```php

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

```

### If you run this code the following things will happen: 

    1. Each image will be examined pixel by pixel and based on it's color, encoded as a 0 (black) or 1 (white). As mentioned in the previously mentioned, I chose this encoding scheme because I wanted to use high contrast images with the text being white and the background being black, adjust as necessary.
    2. Each pattern of 1's and 0's will be assigned a floating point value between 0 and 1.
    3. All the encoded patterns and their values will be saved to a training data file for later use as inputs for the ANN.


Once the images are encoded as numbers representing the pixel color we are now ready to teach our ANN how to identify symbols in images.

## [train_ocr.php](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Basic/train_ocr.php)
```

<?php
set_time_limit ( 300 ); // max run time 5 minutes (adjust as needed)
$num_input = 160;
$num_output = 1;
$num_layers = 3;
$num_neurons_hidden = 107;
$desired_error = 0.00001;
$max_epochs = 5000000;
$epochs_between_reports = 10;
$ann = fann_create_standard($num_layers, $num_input, $num_neurons_hidden, $num_output);
if ($ann) {
	echo 'Training OCR... '; 
	fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC);
	$filename = dirname(__FILE__) . "/ocr.data";
	if (fann_train_on_file($ann, $filename, $max_epochs, $epochs_between_reports, $desired_error))
		fann_save($ann, dirname(__FILE__) . "/ocr_float.net");
	fann_destroy($ann);
}
echo 'All Done! Now run <a href="test_ocr.php">Test OCR</a><br>' . PHP_EOL;

```
### If you run this code the following things will happen:

    1. A standard fully connected 3 layer backward propagating neural network will be created with 160 inputs, and 1 output.
    2. The ANN will be configured to use the <a href="https://en.wikipedia.org/wiki/Sigmoid_function" target="_blank" rel="noopener noreferrer">Sigmoid</a> activation function.
    3. The ANN is trained, saved and dumped from memory.


Now comes the real fun! Taking it out for a test drive! 😛


Before proceeding here is the code we will be considering:
test_ocr.php

## [test_ocr.php](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Basic/test_ocr.php)
```

<style>
	.blue{color:blue;}
	.green{color:green;}
	.red{color:red;}
</style>

<?php
/*
    OCR( 
	    (string) $img, 
	    (char) $expected, 
		(array) $input, 
		(array)$lookup_array, 
		(FANN neural network resource)$ann
	   );
	
	Description: 
	
	With this function I simply try to illustrate how you could test the OCR ANN. 
				
	$img is a string that should be the name to the training image you are reading from. It is
	used to output the image to the browser results.
	
	$expected is a char that you are actually testing for. eg  
	
	$input should be an array of inputs (encoded pixel data)
	
	$lookup_array should be an array of ASCII characters normalized as floating point values in increments of 0.01
	
	$ann should be a FANN neural network resource.
	
	References:   
	   global - http://php.net/language.variables.scope
	   PHP_EOL - http://php.net/manual/en/reserved.constants.php
	   fann_run() - http://php.net/manual/en/function.fann-run.php
	   floor() - http://php.net/manual/en/function.floor.php
	   count() - http://php.net/manual/en/function.count.php
*/
function OCR($img, $expected, $input, $lookup_array, $ann) {
	global $correct; // refer to the non local $correct variable
	$output = ""; 
	
	/* Display image for reference */
	$output .= "Image: <img src='images/$img'><br>" . PHP_EOL;

	// Run the ANN
	$calc_out = fann_run($ann, $input);
	
	$output .= 'Raw: ' .  $calc_out[0] . '<br>' . PHP_EOL;
	$output .= 'Trimmed: ' . floor($calc_out[0]*100)/100 . '<br>' . PHP_EOL;
	$output .= 'Decoded Symbol: ';
	
	/* What did the ANN think it saw? */
	for($i = 0; $i < count($lookup_array); $i++) {
       if( floor($lookup_array[$i][0]*100)/100 == floor($calc_out[0]*100)/100) {
	        $output .= $lookup_array[$i][1] . '<br>' . PHP_EOL;
	        $output .= "Expected: $expected <br>" . PHP_EOL;
	        $output .= 'Result: ';
	        if($expected == $lookup_array[$i][1]){
	        	$output .= '<span class="green">Correct!</span>';
				
				++$correct;
				
	        }else{
	        	$output .= '<span class="red">Incorrect!</span> <a href="train_ocr.php">Retrain OCR</a>';
	        }
		}
	}
	$output .= '<br><br>' . PHP_EOL;
	
	return $output;	
}


$total = 11; // How many images are to be tested
$correct = 0; // The count of how many images were correctly read by the ANN


/* Setup a resource that points to our ANN .net file */
$train_file = (dirname(__FILE__) . '/ocr_float.net');

/* Confirm the ANN exists */
if (!is_file($train_file))
	die('<span class="red">The file ocr_float.net has not been created!</span><a href="train_ocr.php">Train OCR</a>' . PHP_EOL);

/* Create the ANN resource */
$ocr_ann = fann_create_from_file($train_file);
if ($ocr_ann) {
	// Display the images we are testing (hard coded)
	?>
	<h1 class='blue'>OCR Test</h1>
	<strong>Testing: </strong>
	<img src='images/38.png'> <!-- G -->
	<img src='images/68.png'> <!-- e -->
	<img src='images/68.png'> <!-- e -->
	<img src='images/74.png'> <!-- k -->
	<img src='images/38.png'> <!-- G -->
	<img src='images/72.png'> <!-- i -->
	<img src='images/81.png'> <!-- r -->
	<img src='images/75.png'> <!-- l -->
	<img src='images/41.png'> <!-- J -->
	<img src='images/78.png'> <!-- o -->
	<img src='images/88.png'> <!-- y -->
	<br>
	<?php

	/* 
	    Create the lookup_array from ASCII
		https://en.wikipedia.org/wiki/ASCII
		
        start: ascii dec 33 (!) 
        stop:  ascii dec 126 (~) 
    */
	$result_lookup_array = array();
	$curr = 0.00;
	for($i = 33; $i <= 126; $i++) {
		array_push($result_lookup_array, array($curr, chr($i)));
		$curr+= 0.01;
	}
	

	// For simplicity sake I hardcoded these values below as there is no need to prove that we can read
	// the pixel data for each image (as we already did that in generate_training_data.php) however you 
	// can implement similar methodology to what as I did with GenerateTrainingData() to read pixel values
	// programmaticlly into an array rather than manually specifying it as I show here.
	
	$test_G = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_e = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_k = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_i = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_r = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_l = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_J = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_o = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	$test_y = array(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	
       
	/* Test OCR and buffer results in the $output variable as string data */
	$output = "";
	$output .= OCR('38.png', 'G', $test_G, $result_lookup_array, $ocr_ann);
	$output .= OCR('68.png', 'e', $test_e, $result_lookup_array, $ocr_ann);
	$output .= OCR('68.png', 'e', $test_e, $result_lookup_array, $ocr_ann);
	$output .= OCR('74.png', 'k', $test_k, $result_lookup_array, $ocr_ann);
	$output .= OCR('38.png', 'G', $test_G, $result_lookup_array, $ocr_ann);
	$output .= OCR('72.png', 'i', $test_i, $result_lookup_array, $ocr_ann);
	$output .= OCR('81.png', 'r', $test_r, $result_lookup_array, $ocr_ann);
	$output .= OCR('75.png', 'l', $test_l, $result_lookup_array, $ocr_ann);
	$output .= OCR('41.png', 'J', $test_J, $result_lookup_array, $ocr_ann);
	$output .= OCR('78.png', 'o', $test_o, $result_lookup_array, $ocr_ann);
	$output .= OCR('88.png', 'y', $test_y, $result_lookup_array, $ocr_ann);
	
    // Determine how accurate the Neural Network is
    $percent_correct = round(($correct / $total) * 100, 2 );
    
	// Output the accuracy results
	echo "<strong>Results:</strong> $correct images correctly decoded out of $total. (<span class='"; 
	
	/* Add a css style to the percentage results */
	if($percent_correct < 70){echo "red'>";}
	elseif($percent_correct < 90){echo "blue'>";}
	else{echo "green'>";}
	
	/* Close css style span and offer link to retrain */
	echo $percent_correct . "%</span>)<br>Not good enough? <a href='train_ocr.php'>Retrain OCR</a><br><br>" ;
	
	/* display detailed results */
	echo "<h2 class='blue'>Details</h2>";
	echo $output;
	
	/* Free up memory associated with the OCR ANN resource. */ 
	fann_destroy($ocr_ann);
} else {
	die("<span class='red'>Invalid file format.</span>" . PHP_EOL);
}

?>

```

### If you run this code the following things will happen:

    1. The images we are testing will display on page using HTML <img> elements.
    2. Our neural network will be loaded from the file we created in step 3 of this tutorial.
    3. 11 Tests using the OCR function I provide will be preformed.
    4. The results of the ANN will be computed then displayed as an accuracy %.
    5. Detailed results of each test result will be displayed.


Now that we have tested the OCR ANN, lets break it down and understand what is going on.


## Step 1 - generate_training_images.php

In Step 1 we create our images and log file.

![Step 1](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Project%20Images/ocr_step1.png)
...
![Step 1-2](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Project%20Images/ocr_step1-2.png)


## Step 2 - generate_training_data.php

In Step 2 we use the log file as a reference to step through each image and examine every pixel and assign it a value of 1 or 0 based on the color of the pixel. We then save our results to a new file called ocr.data.

![Step 2](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Project%20Images/ocr_step2.png)

## Step 3 - train_ocr.php

In Step 3 we train the neural network and save it as ocr_float.net.

![Step 3](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Project%20Images/ocr_step3.png)

## Step 4 - test_ocr.php

In Step 4 load the ANN from file ocr_float.net and then proceed to test it. In this image I ran multiple tests and excluded the individual image details however in the code I provided you will get additional data about each test image.

![Step 4](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/Project%20Images/ocr_step4.png)

At this point our toy OCR neural network is complete and operating as well as can be expected.

Why is this a "toy" neural network? Because it is trained to "classify" or identify images in a single one off event rather than convolving (yes that is a real word :-P ) over the images and extracting features. Basically what this means is that while it eventually got a ~73% correct identification rate it's not actually going to read text out of just any image... not only that we broke the cardinal rule of testing our ANN with the same data we trained it on (always test on new data not what it was trained on) so the ANN is "hyper fitting" its data set.

We could probably improve accuracy by moving the letters in the images, blurring them rotating them, changing their color and adding the ability for the ANN to work with more than black and white text etc... but in the end it would only be a marginal improvement.

To use OCR for more real world scenarios you will need to implement convolution layers. Which will allow you to not only read the letters and words in an image of any size or color but also do object recognition such as test image 1 is a cat and test image 2 is a ti82 calculator...

And with that I hope you had as much fun following along with this tutorial as much as I had creating it! :-)

** Please support me on [Patreon](https://www.patreon.com/user?u=3969727) so I can bring you more awesome projects! **


**Note:** *This project (all the code, the title images as well as the infographic licensed under everyone's favorite license [MIT LICENSE](https://github.com/geekgirljoy/OCR_Neural_Network/blob/master/LICENSE), though I am rather partial to CC0 :-P, so feel free to take this code and develop it into something amazing! Please just attribute me as the author of the initial code base. Also if you use this to create something cool, I'd love to hear about it!* :-)

As always I hope you found this project both interesting and informative. Please Like, Comment &amp; Share this post with your friends and followers on your social media platforms.

If would like to suggest a topic or project for an upcoming post feel free to [contact me](https://geekgirljoy.wordpress.com/contact/).

Much Love,<br>
~Joy
