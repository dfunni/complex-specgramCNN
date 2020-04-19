# rm logs/*
for f in ./logs/iqmpu*; do
    cp -- "$f" "./csvs/$(basename -- "$f" .log).csv"
done

