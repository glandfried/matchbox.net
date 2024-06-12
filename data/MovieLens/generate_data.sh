mkdir ml-100k
mkdir ml-100k-binary
unzip ml-100k.zip
cd ml-100k && ./mku.sh && cd .. && python generate_100k_csvs.py