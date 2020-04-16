for f in ./logs/*.log; do
    cp -- "$f" "./csv/$(basename -- "$f" .log).csv"
done

