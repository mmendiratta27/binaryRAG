python3 rag.py

DIFF=$(diff statements/expected.txt statements/results.txt) 
if [ "$DIFF" == "" ] 
then
    echo "Expected and Actual Results Match!"
else
    echo "Uh Oh! Some difference between the files"
    echo "$DIFF"
fi