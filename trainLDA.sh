python filterBeforeLDA.py
echo "Filtering done..."
mv newResults.txt LDAAfterStopWordRemoval/
echo "Training starts..."
GibbsLDA++-0.2/src/lda -est -alpha 0.001 -savestep 10 -twords 20 -ntopics 20 -dfile LDAAfterStopWordRemoval/newResults.txt
echo "Training complete... Testing Begins"
GibbsLDA++-0.2/src/lda -inf -dir LDAAfterStopWordRemoval -model model-final -niters 100 -dfile cfile.txt
echo "Everything done!"
