for hw in 0.5 1 2 4; do
    echo "Processing ${hw}s"
    python3 -u testetestetsteyadadfbnaf.py --matrix-half-width "${hw}" > "log_hw${hw}.txt" 2>&1
done