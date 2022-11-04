FROM alpine:3.16.2

ENV MODELS_DIR="/data/models"
ENV MODEL="codegen-350M-mono"
ENV NUM_GPUS=1

RUN apk --no-cache add \
    zstd \
    curl \
    bash

COPY configs/ /data/configs/
COPY load-model.sh .

RUN chmod +x /load-model.sh
ENTRYPOINT ["/load-model.sh"]
CMD ["/data/configs"]