if ! [ -d Results ] ; then
 mkdir Results
fi

cd Data
tar -zxf photo.tar.gz
echo 'Photo Decompression Done.'
tar -zxf sketch.tar.gz
echo 'Sketch Decompression Done.'

mkdir Preprocessed
echo 'Start Data Split...'
python SplitDivision.py
