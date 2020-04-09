# Backend Setup
This guide assumes a mac or linux like environment and if you are on windows then there
may be some different or additional things you have to do.

## Requirements
- if on windows, linux subsystem for windows (LSW) is highly recommended
- python3
- mongodb server
  - alternatively you can set up an account on [atlas](atlas.mongodb.com) and get a connection string
- virtualenv or venv package

## A note about python virtual environments
virtualenv and the env package are tools that help you set up a virtual environment for python.
when you run python or install packages it keeps everything specific to that environment.
so when you install a package in the environment it doesn't install it globally or
dilute your global set up. this helps prevent package conflicts when you work with a lot of
python projects. too see this in action type `which python` outside the environment and then
while inside the environment. you will notes that outside it will look something like
`/usr/local/bin/python` and inside it will look like `env/bin/python`

## Setup
- `cd backend`
- `python3 -m venv env` or `virutalenv -p python3 env`
- `cp cybnetics/config.example.py cybnetics/config.py`
- change any config options you want to

## Running the backend
- `cd backend`
- `source env/bin/activate`
- `python app.py`

## Shutting down and exiting the environment
- press Control+C
- `deactivate`

## debugger
if the server throws an error it should show a screen asking for the debugger pin.
when you started the server it will say what this pin is in the terminal.
if you enter the pin you get access to some fancy features and can poke around the
app at various points kind of like gdb. on top of this pdb and obviously print statements
are also available