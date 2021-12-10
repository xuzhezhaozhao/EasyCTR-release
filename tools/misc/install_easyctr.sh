
set -e

cd git/easyctr/ && git pull && ./pack.sh 
cd -

cp git/easyctr/build/pkg/easyctr.tar.gz .
rm -rf easyctr
tar xvf easyctr.tar.gz
