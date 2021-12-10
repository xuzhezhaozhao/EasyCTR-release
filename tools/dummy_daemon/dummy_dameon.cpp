#include <unistd.h>

int main(int argc, char *argv[]) {
  int rc = daemon(1, 0);
  (void)rc;
  sleep(3153600000);   // aha, 100 years
  return 0;
}
