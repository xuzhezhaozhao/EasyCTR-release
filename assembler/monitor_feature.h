#ifndef ASSEMBLER_MONITOR_FEATURE_H_
#define ASSEMBLER_MONITOR_FEATURE_H_

namespace assembler {

void training_monitor_feature(int idx, int pos1, int pos2);
void serving_monitor_feature(int idx, int pos1, int pos2);

}  // namespace assembler

#endif  // ASSEMBLER_MONITOR_FEATURE_H_
