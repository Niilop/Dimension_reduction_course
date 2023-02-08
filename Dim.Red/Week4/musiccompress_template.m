[originalvector,originalfrequency,originalbits]=wavread("the_entertainer.wav");
nelements=size(originalvector,1)*size(originalvector,2);
blocksize=100;
nblocks=floor(nelements/blocksize);
featuredata=zeros(nblocks,blocksize);
for blockindex=1:nblocks,
  featuredata(blockindex,:)=originalvector( ((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize))';
end;

% -------------------your code begins here----------------

% Insert your code here to compute a PCA reconstruction of the feature data into 
% a new matrix called reconstructed_featuredata

% -------------------your code ends here----------------


reconstructedvector=zeros(nblocks,blocksize);
for blockindex=1:nblocks,
  reconstructedvector(((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize))=(reconstructed_featuredata(blockindex,:))';
end;

wavwrite(reconstructedvector,originalfrequency,originalbits,"pca_musicreconstruction.wav");


