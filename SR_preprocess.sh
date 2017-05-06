# download celebA dataset and unzip
python download_celebA.py
mv celebA celebA.zip
unzip celebA.zip

# crop and resize image to 64x64
# this might take a while
python resize.py
