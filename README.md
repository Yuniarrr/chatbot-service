# Chatbot Service

## Table of Contents

- [How to run this program](#how-to-run-this-program)
- [Tips](#tips)

## How to run this program

1. Create a new venv

```sh
python -m venv .venv
```

Note: This code was built in Python: 3.10

Windows:

```sh
source .venv/Scripts/activate
```

Linux:

```sh
source .venv/bin/activate
```

2. Install dependencies

```sh
pip install -r requirements.txt
```

## How to create migrations

1. Update or add new model in `app/models`

2. Run this command

```sh
alembic revision --autogenerate -m "Name migration"
```

3. Then apply it

```sh
alembic upgrade head
```

4. or, can use this file

```sh
./create-migrations.sh
```

## Git indentity unknown

1. Check username

```sh
echo $USERNAME
```

2. Then

```sh
export HOME=/c/Users/LENOVO
echo 'export HOME=/c/Users/LENOVO' >> ~/.bashrc
source ~/.bashrc
```

## Colbert

```sh
pip install git+https://github.com/stanford-futuredata/ColBERT.git
```

## Tips

1. Change python version: [here](https://stackoverflow.com/questions/70422866/how-to-create-a-venv-with-a-different-python-version)
