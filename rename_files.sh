for f in ./csv/*.csv; do
    mv "$f" "${f/m_pu/mu}"
done
