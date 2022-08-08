# First Obtain Directory Information. The default installation path is always the same as Git Directory
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Required Packages
sudo apt-get update
sudo apt-get install python3-pip libjpeg-dev libjpeg8-dev libpng-dev apache2 libapache2-mod-wsgi-py3 python3-virtualenv libmysqlclient-dev mysql-server

virtualenv $SCRIPT_DIR/PerceptPlatformEnv
source $SCRIPT_DIR/PerceptPlatformEnv/bin/activate
pip3 install django djangorestframework numpy scipy pandas spectrum mysqlclient requests websocket-client joblib

ADMIN_EMAIL='Admin@PerceptPlatform.demo'
ADMIN_PASSWORD=`cat /dev/urandom | tr -dc '[:alpha:]' | fold -w ${1:-20} | head -n 1`
DATASERVER_PATH='/home/ubuntu/Storage/'
SERVER_ADDRESS='0.0.0.0'
[ -z "$SECRET_KEY" ] && SECRET_KEY=`cat /dev/urandom | tr -dc '[:alpha:]' | fold -w ${1:-64} | head -n 1`
export SECRET_KEY=$SECRET_KEY

# Ask User for Admin Email Address
read -p "Please Enter Admin Email Address [default: 'Admin@PerceptPlatform.demo']: " userInput
[ ! -z "$userInput" ] && ADMIN_EMAIL=$userInput

# Ask User for Admin Email Address
read -sp "Password: " userInput
[ ! -z "$userInput" ] && ADMIN_PASSWORD=$userInput
echo

# Ask User for Installation Path
read -p "Please Enter Storage Path [default: '/home/ubuntu/Storage/']: " userInput
[ ! -z "$userInput" ] && DATASERVER_PATH=$userInput

# Ask User for Domain IP
read -p "Please Enter Domain IP [default: '0.0.0.0']: " userInput
[ ! -z "$userInput" ] && SERVER_ADDRESS=$userInput

# Handles the WSGI File Creation
cp $SCRIPT_DIR/PerceptServer_Pro/wsgi.py $SCRIPT_DIR/PerceptServer_Pro/wsgi_production.py
export PERCEPT_DIR="$SCRIPT_DIR/modules/PerceptProcessingModules"
export STATIC_ROOT="$SCRIPT_DIR/PerceptServer_Pro/static"
export DATASERVER_PATH=$DATASERVER_PATH
export SERVER_ADDRESS=$SERVER_ADDRESS
sed -i "14 i os.environ[\"PERCEPT_DIR\"] = \"$PERCEPT_DIR\"" "$SCRIPT_DIR/PerceptServer_Pro/wsgi_production.py"
sed -i "15 i os.environ[\"STATIC_ROOT\"] = \"$STATIC_ROOT\"" "$SCRIPT_DIR/PerceptServer_Pro/wsgi_production.py"
sed -i "16 i os.environ[\"SECRET_KEY\"] = \"$SECRET_KEY\"" "$SCRIPT_DIR/PerceptServer_Pro/wsgi_production.py"
sed -i "17 i os.environ[\"SERVER_ADDRESS\"] = \"$SERVER_ADDRESS\"" "$SCRIPT_DIR/PerceptServer_Pro/wsgi_production.py"
sed -i "17 i os.environ[\"DATASERVER_PATH\"] = \"$DATASERVER_PATH\"" "$SCRIPT_DIR/PerceptServer_Pro/wsgi_production.py"

# Create Apache2 Configuration
cp $SCRIPT_DIR/perceptplatform.conf $SCRIPT_DIR/perceptplatform_production.conf
sed -i "s#%{DOCUMENTROOT}#$SCRIPT_DIR#g" $SCRIPT_DIR/perceptplatform_production.conf
sed -i "s#%{ADMIN_NAME}#$ADMIN_EMAIL#g" $SCRIPT_DIR/perceptplatform_production.conf
sed -i "s#%{SERVER_NAME}#$SERVER_ADDRESS#g" $SCRIPT_DIR/perceptplatform_production.conf
cp $SCRIPT_DIR/perceptplatform_production.conf /etc/apache2/sites-available/perceptplatform.conf
a2dissite 000-default.conf
a2ensite perceptplatform.conf
systemctl reload apache2

python3 $SCRIPT_DIR/manage.py makemigrations PerceptDashboard
python3 $SCRIPT_DIR/manage.py migrate
python3 $SCRIPT_DIR/manage.py collectstatic

mkdir $DATASERVER_PATH
mkdir $DATASERVER_PATH/cache
mkdir $DATASERVER_PATH/sessions
mkdir $DATASERVER_PATH/recordings
