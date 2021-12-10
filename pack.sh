
set -e

bash build.sh

PKG=build/pkg/easyctr

rm -rf ${PKG}

mkdir -p ${PKG}

cp -r easyctr ${PKG}
cp -r common ${PKG}

mkdir -p ${PKG}/tools/string_indexer
mkdir -p ${PKG}/tools/conf_generator
mkdir -p ${PKG}/ops/
cp ./build/tools/string_indexer/string_indexer ${PKG}/tools/string_indexer
cp ./tools/conf_generator/conf_generator.py ${PKG}/tools/conf_generator/
cp -r ./tools/train ${PKG}/tools/
cp -r ./tools/misc ${PKG}/tools/
cp -r ./tools/dump_params ${PKG}/tools/
cp -r ./tools/adagrad_shrinkage ${PKG}/tools/

cp main.py ${PKG}
cp ./build/ops/libassembler_ops.so ${PKG}/ops/

find ${PKG} -name *.pyc -exec rm {} \;

cd ${PKG}/..
tar cvzf easyctr.tar.gz easyctr
