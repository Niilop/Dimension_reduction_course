library("tuneR");
original<-readWave("the_entertainer.wav");
originalvector<-as.matrix(original@left);
nelements<-dim(originalvector)[1]*dim(originalvector)[2];
blocksize<-100;
nblocks<-floor(nelements/blocksize);
featuredata<-matrix(0,nblocks,blocksize);
for (blockindex in 1:nblocks)
{
  featuredata[blockindex,]<-t(originalvector[((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize)]);
}

# -------------------your code begins here----------------

# Insert your code here to compute a PCA reconstruction of the feature data into 
# a new matrix called reconstructed_featuredata

# -------------------your code ends here----------------


reconstructedvector<-matrix(0,nblocks*blocksize,1);
for (blockindex in 1:nblocks)
{
  reconstructedvector[((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize)]<-t(reconstructed_featuredata[blockindex,]);
}

reconstructedaudio<-original;
reconstructedaudio@left<-as.vector(reconstructedvector);

writeWave(reconstructedaudio,"pca_musicreconstruction.wav");
