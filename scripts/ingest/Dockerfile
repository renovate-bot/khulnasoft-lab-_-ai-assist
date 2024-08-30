FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    wget \
    llvm \
    ca-certificates \
    # For asdf Python https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    # For asdf Ruby https://github.com/rbenv/ruby-build/wiki#suggested-build-environment
    autoconf patch rustc libyaml-dev libreadline6-dev libgmp-dev libgdbm6 libgdbm-dev libdb-dev uuid-dev

# Set the locale UTF8
ENV LANG C.UTF-8
ENV LANGUAGE C
ENV LC_ALL C.UTF-8

WORKDIR /app

# Install runtime via .tool-versions of asdf
ENV PATH="/root/.asdf/bin:/root/.asdf/shims:$PATH"
COPY .tool-versions /app/.tool-versions
RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.0

# Set up Python first since gcloud is not yet compatible with Python 3.12:
# https://issuetracker.google.com/issues/303737178?pli=1
RUN asdf plugin-add python
RUN asdf install python
RUN ln -sf /root/.asdf/shims/python /root/.asdf/shims/python3

RUN asdf plugin-add poetry https://github.com/asdf-community/asdf-poetry
RUN asdf plugin-add ruby https://github.com/asdf-vm/asdf-ruby
RUN asdf plugin-add gcloud https://github.com/jthegedus/asdf-gcloud
RUN asdf plugin add vale https://github.com/pdemagny/asdf-vale
RUN asdf plugin add markdownlint-cli2 https://github.com/paulo-ferraz-oliveira/asdf-markdownlint-cli2
RUN asdf plugin add nodejs https://github.com/asdf-vm/asdf-nodejs
RUN asdf plugin add lefthook https://github.com/jtzero/asdf-lefthook

# Install nodejs first since markdownlint-cli2 needs it.
# This can be dropped once asdf installs dependencies in order: https://github.com/asdf-vm/asdf/pull/1723
RUN asdf install nodejs

RUN asdf install

# Install python dependencies
ENV VIRTUAL_ENV=/usr/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY poetry.lock pyproject.toml /app/
RUN python -m venv $VIRTUAL_ENV && poetry install --no-interaction --no-ansi --no-dev

# Copy scripts
COPY Makefile /app/Makefile
COPY scripts /app/scripts

CMD ["make", "ingest"]
