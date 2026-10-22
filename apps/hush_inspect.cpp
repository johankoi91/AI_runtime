#include <cstdlib>
#include <iostream>

#include "runtime/hush_model.h"

int main() {
  const char* path = std::getenv("HUSH_MODEL_BUNDLE");
  if (path == nullptr || *path == '\0') {
    std::cout << "HUSH_MODEL_BUNDLE is not set; real model inspection skipped\n";
    return 0;
  }

  try {
    const runtime::HushModel model = runtime::HushModel::load_for_inspection(path);
    const runtime::HushGraphReport reports[] = {
        runtime::inspect_hush_graph(model.encoder),
        runtime::inspect_hush_graph(model.erb_decoder),
        runtime::inspect_hush_graph(model.df_decoder),
    };
    const char* names[] = {"encoder", "erb_decoder", "df_decoder"};
    for (int index = 0; index < 3; ++index) {
      std::cout << names[index] << ": nodes=" << reports[index].node_count << " ops=";
      for (std::size_t op_index = 0; op_index < reports[index].op_types.size(); ++op_index) {
        if (op_index > 0) std::cout << ',';
        std::cout << reports[index].op_types[op_index];
      }
      std::cout << "\n";
      if (!reports[index].unsupported_ops.empty()) {
        std::cout << "  unsupported=";
        for (std::size_t op_index = 0; op_index < reports[index].unsupported_ops.size(); ++op_index) {
          if (op_index > 0) std::cout << ',';
          std::cout << reports[index].unsupported_ops[op_index];
        }
        std::cout << "\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Hush model inspection failed: " << error.what() << "\n";
    return 1;
  }
}
