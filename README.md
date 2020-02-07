# Flask Face Recognition
A face recognition API using Python, Flask, Opencv, Pytorch, Heroku.

## Sign up for Heroku

Heroku, being a Software as a Service (SaaS)-type of service, requires you to create an account and login before you can start using its computers. Don't worry, creating an account and running a simple app is free and doesn't require a credit card.

You can create an account at this URL: [https://signup.heroku.com/](https://signup.heroku.com/)

## Download the Heroku toolbelt

Heroku has a command-line "toolbelt" that we must download and install in order commands that will simplify our communication with the Heroku servers. The toolbelt can be downloaded at: [https://toolbelt.heroku.com/](https://toolbelt.heroku.com/)


## Authenticate with Heroku with `heroku login`

Installing the Heroku toolbelt will give you access to the `heroku` command which has several subcommands for interacting with the Heroku service. 

The first command you need to run is `heroku login`, which will ask you to enter your login credentials so that every subsequent `heroku` command knows who you are:

(You will only have to do this __once__)

~~~
heroku login
~~~

## Update `app.py` for your needs

Example app.py file includes basic Flask API structure, populate it with your API methods:

~~~py
from flask import Flask,jsonify,request,render_template

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
~~~

You should be able to run this app on your own system via the familiar invocation and visiting [http://localhost:5000](http://localhost:5000):

~~~
python app.py
~~~

## Specifying dependencies for deploying Heroku

(the following comes from Heroku's guide to [Deploying Python and Django Apps on Heroku](https://devcenter.heroku.com/articles/deploying-python))

Our simple Flask app has has a couple of __dependencies__: the Python interpreter and the Flask library. Which means that Python and the Flask library must be installed on our computer.

With Heroku, we have to include some metadata with our application code, so that Heroku knows how to set up a compatible webserver and install the software that our application needs. The metadata can be as simple as including a few plaintext files, which I list below in the next section. 

### Installing the gunicorn web server

Whenever we run `python app.py` from our command-line, we're running the default webserver that comes with Flask. However, Heroku seems to prefer a [web server called __gunicorn__](https://devcenter.heroku.com/articles/python-gunicorn). Just so that we can follow along with Heroku's documentation, let's install gunicorn on our own system. It's a Python library like any other and can be installed with __pip__:

~~~
pip install gunicorn
~~~

### Adding a requirements.txt

By convention, Python packages often include a plaintext file named __requirements.txt__, in which the dependencies for the package are listed on each line.

Create an empty file named `requirements.txt` (in the same root folder as __app.py__) and add our dependencies on it.

~~~
flask
gunicorn
~~~

### Specifying Python version with `runtime.txt`

Heroku will know that we be running a Python app, but because there's a huge disparity between Python versions (notably, [Python 2 versus 3](https://wiki.python.org/moin/Python2orPython3)), we need to tell Heroku to use the Python version that we're using on our own computer to develop our app.

Which version of Python are we/you running? From your command line, run Python and note the Python version:

~~~
python
~~~

~~~
Python 3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 14:00:49) on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> 
~~~

Nevermind that `"Anaconda"` bit -- we just need the version number, e.g. __3.6.9__. Then exit Python with

~~~
exit()
~~~

Create __runtime.txt__ in your root app folder and add just the single line (note: replace my example version number with yours, if it is different):

~~~
python-3.6.9
~~~

## Create a `Procfile`

OK, one more necessary plaintext file: [Heroku needs a file to tell it how to start up the web app](https://devcenter.heroku.com/articles/deploying-python#the-procfile). By convention, this file is just a plaintext file is called (and named): __Procfile__:

> A Procfile is a text file in the root directory of your application that defines process types and explicitly declares what command should be executed to start your app. 


And for now, __Procfile__ can contain just this line:

~~~
web: gunicorn app:app --log-file=-
~~~
