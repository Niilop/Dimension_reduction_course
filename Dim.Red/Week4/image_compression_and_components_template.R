#=====================================
# This template shows how principal component analysis (PCA) 
# can be used for image compression.
# Created for the course DATA.STAT.770, author Jaakko Peltonen, 2021.
#=====================================

# Read the example image. The result will be a 3D matrix (pixel-rows * pixel-columns * color-channels)
# If you do not have the png library installed, you can install it with the command install.packages("png")
library(png);
print("Reading in the image");
imagematrix<-readPNG("rms_olympic_grand_staircase.png");

# Reduce the image to grayscale, by taking the average value of the red, green, and blue color-channels.
# The result will be a 2D matrix (pixel-rows * pixel-columns)
print("Reducing the image to grayscale");
imagematrix<-(imagematrix[,,1]+imagematrix[,,2]+imagematrix[,,3])/3

# Find out how many pixel-rows and pixel-columns there are.
npixelrows<-nrow(imagematrix);
npixelcolumns<-ncol(imagematrix);

# Draw the original image on-screen with a grayscale colormap
library(graphics)
print("Drawing the image");
image(1:npixelcolumns,1:npixelrows,t(imagematrix),col=gray.colors(256),ylim=c(npixelrows,1));
title("Original image");

# Divide the image into 10x10 pixel blocks (100 pixels in each block)
blocksize_pixelrows<-10;
blocksize_pixelcolumns<-10;

# Find out how many rows and columns of 10x10 blocks fit inside the image.
nblockrows<-floor(npixelrows/blocksize_pixelrows);
nblockcolumns<-floor(npixelcolumns/blocksize_pixelcolumns);


# Create a new matrix "featuredata" whose rows are the blocks (taken block-column by block-column, block-row by block-row from the image)
# and whose columns are the 100 pixels of each block (taken pixel-column by pixel-column, pixel-row by pixel-row from each block).
# The resulting "featuredata" matrix can be seen as a data set, where each block is a data item, and each block has a 100-dimensional
# feature vector of pixel values.
print("Creating the featuredata matrix");
featuredata<-matrix(0,nblockcolumns*nblockrows,blocksize_pixelcolumns*blocksize_pixelrows);

# Collect the pixel values in all blocks into the featuredata matrix
featurerow_index <- 0;
for (blockcolumn_index in 1:nblockcolumns)    # Go over each block column
{
  for (blockrow_index in 1:nblockrows)        # Go over each block row
  {
    featurerow_index<-featurerow_index+1;  # Row-index of this block in the featuredata matrix

    # Compute the indices of the first and last pixel-columns in this block, and the first and last pixel-rows in this block
    first_pixelcolumn <- (blockcolumn_index-1)*blocksize_pixelcolumns + 1;
    last_pixelcolumn <- (blockcolumn_index-1)*blocksize_pixelcolumns + blocksize_pixelcolumns;
    first_pixelrow <- (blockrow_index-1)*blocksize_pixelrows + 1;
    last_pixelrow <- (blockrow_index-1)*blocksize_pixelrows + blocksize_pixelrows;

    # Extract the pixels belonging to the block from the image
    blockpixels<-imagematrix[first_pixelrow:last_pixelrow,first_pixelcolumn:last_pixelcolumn];

    # Put the pixels column-by-column, row-by-row into the current row of the featuredata matrix
    featuredata[featurerow_index,]<-t(blockpixels[1:(blocksize_pixelrows*blocksize_pixelcolumns)]);
  }
}


#==============================INSERT YOUR CODE BELOW=====================

# Insert your own code between these lines to solve this exercise. 
#
# Your code should create a new matrix "reconstructed_featuredata" which is the same size as "featuredata", 
# but the feature values in each row have been reconstructed from a low-dimensional PCA projection of the original
# 100-dimensional featuredata.
#
# Your code should also create a new matrix "principalcomponents" which has the same number of columns as 
# "featuredata", and has as many rows as you have principal components. Each row of this matrix should contain 
# the principal component projection direction for one of the principal components.
#
# The rest of the code below these lines assumes that the "reconstructed_featuredata" matrix and the 
# "principalcomponents" matrix have been created. Note that the code draws two images: you will need to
# press the Enter key to continue from the first image to the second.

#==============================INSERT YOUR CODE ABOVE=====================


# Create a reconstructed image based on the "reconstructed_featuredata" matrix.
print("Creating the reconstructed image");
reconstructed_imagematrix <- matrix(0,npixelrows,npixelcolumns);

featurerow_index <- 0;
for (blockcolumn_index in 1:nblockcolumns)    # Go over each block column
{
  for (blockrow_index in 1:nblockrows)        # Go over each block row
  {
    featurerow_index<-featurerow_index+1;  # Row-index of this block in the reconstructed_featuredata matrix

    # Create a pixel block of the correct size
    blockpixels<-matrix(0,blocksize_pixelrows, blocksize_pixelcolumns);

    # Take the pixels column-by-column, row-by-row from the current row of the reconstructed_featuredata matrix
    blockpixels[1:(blocksize_pixelrows*blocksize_pixelcolumns)] <- reconstructed_featuredata[featurerow_index,];

    # Compute the indices of the first and last pixel-columns in this block, and the first and last pixel-rows in this block
    first_pixelcolumn <- (blockcolumn_index-1)*blocksize_pixelcolumns + 1;
    last_pixelcolumn <- (blockcolumn_index-1)*blocksize_pixelcolumns + blocksize_pixelcolumns;
    first_pixelrow <- (blockrow_index-1)*blocksize_pixelrows + 1;
    last_pixelrow <- (blockrow_index-1)*blocksize_pixelrows + blocksize_pixelrows;

    # Place the pixels belonging to the block into the correct position in the reconstructed image
    reconstructed_imagematrix[first_pixelrow:last_pixelrow,first_pixelcolumn:last_pixelcolumn] <- blockpixels;
  }
}

# Show the reconstructed image
print("Drawing the reconstructed image");
image(1:npixelcolumns,1:npixelrows,t(reconstructed_imagematrix),col=gray.colors(256),ylim=c(npixelrows,1));
title("Reconstructed image");

readline(prompt="Press the Enter key to continue");

# Create an image of the principal component projection directions based on the "principalcomponents" matrix.
print("Creating the principal components image");
ncomponents<-dim(principalcomponents)[1];
principalcomponents_imagematrix <- matrix(0,blocksize_pixelrows,ncomponents*blocksize_pixelcolumns);

featurerow_index <- 0;
for (component_index in 1:ncomponents)    # Go over each component
{
  # Create a pixel block of the correct size
  blockpixels<-matrix(0,blocksize_pixelrows, blocksize_pixelcolumns);

  # Take the pixels column-by-column, row-by-row from the current row of the principal component matrix
  blockpixels[1:(blocksize_pixelrows*blocksize_pixelcolumns)] <- principalcomponents[component_index,];

  # Compute the indices of the first and last pixel-columns in this block, and the first and last pixel-rows in this block
  first_pixelcolumn <- (component_index-1)*blocksize_pixelcolumns + 1;
  last_pixelcolumn <- (component_index-1)*blocksize_pixelcolumns + blocksize_pixelcolumns;
  first_pixelrow <- 1;
  last_pixelrow <- blocksize_pixelrows;

  # Place the pixels belonging to the block into the correct position in the reconstructed image
  principalcomponents_imagematrix[first_pixelrow:last_pixelrow,first_pixelcolumn:last_pixelcolumn] <- blockpixels;
}

# Show the reconstructed image
print("Drawing the principal components image");
image(1:dim(principalcomponents_imagematrix)[2],1:dim(principalcomponents_imagematrix)[1],t(principalcomponents_imagematrix),col=gray.colors(256),ylim=c(blocksize_pixelrows,1));
title("Principal components of the image");

