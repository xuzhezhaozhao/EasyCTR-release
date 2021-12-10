#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <sys/stat.h>

#include <boost/foreach.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "thirdparty/gflags/gflags.h"
#include "thirdparty/glog/logging.h"

#include "tensorflow/contrib/makefile/gen/proto/tensorflow/core/example/example.pb.h"
#include "tensorflow/contrib/makefile/gen/proto/tensorflow/core/example/feature.pb.h"

#include "service/server/algorithm/algorithmmanager.h"
#include "service/server/algorithm/algorithm_plugins/pctr_tf_dnn_dlrm_long/tf_dnn_factory.h"
#include "service/server/algorithm/algorithm_plugins/pctr_tf_dnn_dlrm_long/tf_dnn_predictor.h"
#include "service/util/seclimit.h"
#include "service/util/string_split.h"
#include "service/util/toolfun.h"
#include "service/util/zlib_util.h"

namespace video_rec_engine {

namespace algorithm_pctr_tf_dnn_dlrm_long {

using namespace std;
using namespace tensorflow;
using namespace boost::property_tree;

TFDNNPredictor::~TFDNNPredictor() {
    if (m_bundle.session != NULL) {
        tensorflow::Status status = m_bundle.session->Close();
        if (!status.ok()) {
            LOG(ERROR) << "close seesion failed, algid " << m_algid.toString()
                << ". status " << status;
        }
    }
    VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
        << m_file_name << " deconstrcuct sucess.";
}

bool TFDNNPredictor::CheckDelete() {
    return true;
}

bool TFDNNPredictor::Delete() {
    return true;
}

void TFDNNPredictor::SetAlgID(AlgID &algid) {
    m_algid = algid;
    return;
}

bool TFDNNPredictor::LoadPredictor(const string &data_file, bool slow) {
    struct stat statbuf;
    stat(data_file.c_str(), &statbuf);
    if (statbuf.st_size == 0) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " file " << data_file
            << " is empty.";
        return false;
    }
    m_file_size = statbuf.st_size;
    m_file_name = data_file;

    if (!loadModelFromZip(m_file_name, slow)) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " load model from " << data_file
            << " file size " << m_file_size
            << " failed.";
        return false;
    }

    warmupModel();
    return true;
}

bool TFDNNPredictor::loadConf(const std::string &conf_file) {
    if (access(conf_file.c_str(), F_OK) != 0) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << "  config file " << conf_file << " not exist.";
        return false;
    }
    try {
        ptree pt;
        read_xml(conf_file, pt);

        BOOST_FOREACH(ptree::value_type & value, pt.get_child("root")) {
            if (value.first == "intra_op_threads") {
                m_intra_op_threads = value.second.get<int>(
                    "<xmlattr>.number", 10);
                VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
                    << " get intra_op_threads " << m_intra_op_threads;
            } else if (value.first == "inter_op_threads") {
                m_inter_op_threads = value.second.get<int>(
                    "<xmlattr>.number", 10);
                VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
                    << " get inter_op_threads " << m_inter_op_threads;
            } else if (value.first == "model_input") {
                BOOST_AUTO(nextchild, value.second.get_child(""));
                for (BOOST_AUTO(nextpos, nextchild.begin());
                     nextpos != nextchild.end(); ++nextpos) {
                    if(nextpos->first == "tensor") {
                        TFDNNPredictor::TensorInfo tensor_info;
                        tensor_info.name = nextpos->second.get<string>(
                            "<xmlattr>.name", "");
                        string shape_str = nextpos->second.get<string>(
                            "<xmlattr>.shape", "");
                        if (!parseTensorShape(shape_str, tensor_info.shape)) {
                            continue;
                        }
                        tensor_info.desc = nextpos->second.get<string>(
                            "<xmlattr>.desc", "");
                        if (m_input_tensors.insert(make_pair(
                                tensor_info.name, tensor_info)).second) {
                            m_input_tensor_names.push_back(tensor_info.name);
                            VLOG(1) << "tf dnn predictor"
                                << " algid " << m_algid.toString()
                                << " get a input tensor info"
                                << " name " << tensor_info.name << ","
                                << " shape " << tensor_info.shape.DebugString();
                        }
                    }
                }
            } else if (value.first == "model_output") {
                BOOST_AUTO(nextchild, value.second.get_child(""));
                for (BOOST_AUTO(nextpos, nextchild.begin());
                     nextpos != nextchild.end(); ++nextpos) {
                    if (nextpos->first == "tensor") {
                        TFDNNPredictor::TensorInfo tensor_info;
                        tensor_info.name = nextpos->second.get<string>(
                            "<xmlattr>.name", "");
                        string shape_str = nextpos->second.get<string>(
                            "<xmlattr>.shape", "");
                        if (!parseTensorShape(shape_str, tensor_info.shape)) {
                            continue;
                        }
                        tensor_info.desc = nextpos->second.get<string>(
                            "<xmlattr>.desc", "");
                        if (m_output_tensors.insert(make_pair(
                                tensor_info.name, tensor_info)).second) {
                            m_output_tensor_names.push_back(tensor_info.name);
                            VLOG(1) << "tf dnn predictor"
                                << " algid " << m_algid.toString()
                                << " get a output tensor info"
                                << " name " << tensor_info.name << ","
                                << " shape " << tensor_info.shape.DebugString();
                        }
                    }
                }
            }
        }
    } catch (exception &e) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
                << " parse conf file from " << m_file_name
                << " failed " << e.what() << ".";
        return false;
    }
    if (m_input_tensors.empty() || m_output_tensors.empty()) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " load tensor info failed, "
            << " algid " << m_algid.toString()
            << " input tensor size " << m_input_tensors.size()
            << " output tensor size " << m_output_tensors.size();
        return false;
    }
    return true;
}

bool TFDNNPredictor::loadModelFromZip(string &infile, bool slow) {
    // extract zip
    size_t pos = infile.rfind(".");
    if (pos == string::npos) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " model filename " << infile
            << " is invalid.";
        return false;
    }
    string model_dir = infile.substr(0, pos);
    if (uncompress_file(infile, model_dir) < 0) {
        LOG(ERROR) << "zip file " << infile
            << " uncompress failed.";
        slow_del_dir(model_dir.c_str(), slow);
        return false;
    }
    if (!tensorflow::MaybeSavedModelDirectory(model_dir)) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " model directory " << model_dir
            << " does not contain a savedModel.";
        slow_del_dir(model_dir.c_str(), slow);
        return false;
    }

    // load model
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto& config = options.config;
    config.set_intra_op_parallelism_threads(24);
    config.set_inter_op_parallelism_threads(24);
    // cost model config
    // GraphOptions *graph_option = config.mutable_graph_options();
    // graph_option->set_build_cost_model(100);
    // graph_option->set_build_cost_model_after(5);
    VLOG(1) << "algid " << m_algid.toString()
        << " ConfigProto DebugString: " << config.DebugString();

    tensorflow::Status status;
    const std::unordered_set<string> tags = {tensorflow::kSavedModelTagServe};
    status = tensorflow::LoadSavedModel(
        options, tensorflow::RunOptions(),
        model_dir, tags, &m_bundle);
    if (!status.ok()) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " lode model failed, status " << status;
        slow_del_dir(model_dir.c_str(), slow);
        return false;
    }

    // initialize lookup table
    // status = m_bundle.session->Run({}, {}, {"init_all_tables"}, {});
    // if (!status.ok()) {
    //     VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
    //         << " session initialize error " << status;
    //     //return false;
    // }
    // VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
    //     << " load model & init " << model_dir
    //     << " success.";

    // load model_conf
    string conf_file_path = model_dir  + "/model_conf.conf";
    if(!loadConf(conf_file_path)) {
        VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
            << " load model conf file " << conf_file_path
            << " fail.";
        slow_del_dir(model_dir.c_str(), slow);
        return false;
    }
    VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
        << " load conf file " << conf_file_path
        << " success.";

    // cleanup
    slow_del_dir(model_dir.c_str(), slow);
    return true;
}

bool TFDNNPredictor::parseTensorShape(string &str,
                                      tensorflow::TensorShape &shape) {
    string tmpStr;
    tmpStr.assign(str.data(), str.size());
    char sep[2] = ",";
    vector<string> vec_temp;
    StringSplitToVector((char*) tmpStr.data(), tmpStr.size(), sep, strlen(sep),
                        vec_temp, 10);
    if (vec_temp.size() < 1) {
        VLOG(1) << "tf dnn predictor algid " << m_algid.toString()
            << " parse tensor shape str " << str
            << " get dim size " << vec_temp.size();
        return false;
    }
    for (auto &it : vec_temp) {
        shape.AddDim(atoi(it.c_str()));
    }
    return true;
}

bool TFDNNPredictor::PreInitialize() {
    warmupModel();
    return true;
}

void TFDNNPredictor::warmupModel() {
    tensorflow::Example example;
    auto &features = *(example.mutable_features()->mutable_feature());

    features["cid"].mutable_bytes_list()->add_value("w2onnrty6jz3byq");
    features["day_of_week"].mutable_bytes_list()->add_value("6");
    features["hour_of_day"].mutable_bytes_list()->add_value("12");
    features["src_key"].mutable_bytes_list()->add_value("100113");
    features["idx_alg"].mutable_bytes_list()->add_value("5192");
    features["rec_scene"].mutable_bytes_list()->add_value("2");
    features["idx_score"].mutable_float_list()->add_value(0.5);
    features["dev_type"].mutable_bytes_list()->add_value("5");
    features["lbs_province"].mutable_bytes_list()->add_value("156095");
    features["lbs_city"].mutable_bytes_list()->add_value("445100");
    features["age"].mutable_bytes_list()->add_value("32");
    features["gender"].mutable_bytes_list()->add_value("1");
    features["prof"].mutable_bytes_list()->add_value("0");
    features["edu"].mutable_bytes_list()->add_value("0");
    features["prev_watch_cnt"].mutable_int64_list()->add_value(1);
    features["prev_imp_cnt"].mutable_int64_list()->add_value(1);
    features["prev_watch_cnt_since_imp"].mutable_int64_list()->add_value(1);
    features["prev_imp_cnt_since_watch"].mutable_int64_list()->add_value(1);
    features["rec_reason"].mutable_bytes_list()->add_value("");
    features["province"].mutable_bytes_list()->add_value("0");
    features["city"].mutable_bytes_list()->add_value("0");
    features["is_vip"].mutable_bytes_list()->add_value("0");
    features["plat_bucket"].mutable_bytes_list()->add_value("50600");
    features["plat_strategy"].mutable_bytes_list()->add_value("0");
    features["dev_model"].mutable_bytes_list()->add_value("iphone xs");
    features["cid_playlist"].mutable_bytes_list()->add_value("cid1xxxxx");
    features["cid_playlist"].mutable_bytes_list()->add_value("cid2xxxxx");
    features["cid_playlist_cnts"].mutable_int64_list()->add_value(1);
    features["cid_playlist_cnts"].mutable_int64_list()->add_value(2);
    features["cid_playlist_secs"].mutable_int64_list()->add_value(1);
    features["cid_playlist_secs"].mutable_int64_list()->add_value(2);
    features["play_cates"].mutable_bytes_list();
    features["play_cates_cnts"].mutable_int64_list();
    features["play_cates_secs"].mutable_int64_list();
    features["play_gens"].mutable_bytes_list();
    features["play_gens_cnts"].mutable_int64_list();
    features["play_gens_secs"].mutable_int64_list();
    features["play_subgens"].mutable_bytes_list();
    features["play_subgens_cnts"].mutable_int64_list();
    features["play_subgens_secs"].mutable_int64_list();
    features["play_om_firsts"].mutable_bytes_list()->add_value("12");
    features["play_om_firsts_cnts"].mutable_int64_list()->add_value(0);
    features["play_om_firsts_secs"].mutable_int64_list()->add_value(2);
    features["play_om_seconds"].mutable_bytes_list();
    features["play_om_seconds_cnts"].mutable_int64_list();
    features["play_om_seconds_secs"].mutable_int64_list();

    string input_str;
    example.SerializeToString(&input_str);
    tensorflow::Tensor input(tensorflow::DT_STRING,
                             tensorflow::TensorShape({1}));
    input.flat<string>()(0) = input_str;
    VLOG(3) << "tf dnn predictor algid " << m_algid.toString()
        << " warmup case predict one item "
        << " debug info: " << example.DebugString();

    vector<pair<string, tensorflow::Tensor> > in_tensor;
    in_tensor.reserve(1);
    in_tensor.emplace_back("example", input);

    vector<string> out_tensor_name;
    out_tensor_name.push_back("predict");
    vector<tensorflow::Tensor> output_tensors;
    tensorflow::Status status;
    status = m_bundle.session->Run(
        in_tensor, out_tensor_name, {}, &output_tensors);
    if (!status.ok()) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " warm up session run error " << status;
    }
}

bool TFDNNPredictor::Predict(const string &request_id, uint32_t video_num,
                             vector<pair<string, tensorflow::Tensor> > &input,
                             vector<string> &out_tensor_name,
                             vector<tensorflow::Tensor> &output) {
    if (input.empty()) {
        VLOG(1) << "tf dnn predictor input is empty.";
        return false;
    }
    tensorflow::Status status;

    // tensorflow::RunOptions options;
    // options.set_trace_level(RunOptions_TraceLevel_FULL_TRACE);
    // tensorflow::RunMetadata meta_data;
    // status = m_bundle.session->Run(options, input, out_tensor_name, {}, &output, &meta_data);
    status = m_bundle.session->Run(input, out_tensor_name, {}, &output);
    if (!status.ok()) {
        LOG(ERROR) << "tf dnn predictor algid " << m_algid.toString()
            << " predictor session run error " << status;
        return false;
    }

    // string out_str;
    // meta_data.step_stats().SerializeToString(&out_str);
    // string file_name = "profile-" + request_id + "-" + to_string(video_num);
    // std::ofstream ofs(file_name);
    // ofs << out_str;
    // ofs.close();

    return true;
}

} // namespace algorithm_pctr_tf_dnn_dlrm_long

} // namespace video_rec_engine
