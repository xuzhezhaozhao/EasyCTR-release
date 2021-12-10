
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Assembler")
    .Input("input: string")
    .Input("is_predict: bool")
    .Output("feature: double")
    .Output("label: double")
    .Output("weight: double")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Attr("use_lua_sampler: bool = false")
    .Attr("lua_sampler_script: string = ''")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerScheme")
    .Output("scheme: string")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerSerialize")
    .Output("output: string")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerServing")
    .Input("user_feature: string")
    .Input("item_features: string")
    .Output("features: double")
    .SetIsStateful()
    .Attr("serialized: string")
    .Doc(R"doc(
)doc");

// DSSM recall 模式
REGISTER_OP("AssemblerDssmServing")
    .Input("user_feature: string")
    .Output("features: double")
    .SetIsStateful()
    .Attr("serialized: string")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerWithNegativeSampling")
    .Input("input: string")
    .Output("features: double")
    .Output("labels: double")
    .Output("weights: double")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Attr("nce_items_path: string = ''")
    .Attr("nce_count: int = 5")
    .Attr("nce_items_min_count: int = 10")
    .Attr("nce_items_top_k: int = -1")
    .Attr("use_lua_sampler: bool = false")
    .Attr("lua_sampler_script: string = ''")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
