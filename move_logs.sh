rm logs/*
for f in ./csvs/*.csv; do
    cp -- "$f" "./logs/$(basename -- "$f" .csv).log"
done

