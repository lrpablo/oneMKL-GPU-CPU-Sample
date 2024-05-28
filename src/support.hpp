/**
 * Prints the initial values of a matrix.
 *
 * @param M The matrix to print.
 * @param ldM The leading dimension of the matrix.
 * @param MatrixName The name of the matrix.
 */
template <typename T> void print_values(T M, int ldM, std::string MatrixName) {
  std::cout << std::endl;
  std::cout << "\t" << MatrixName << " = [ " << M[0 * ldM] << ", " << M[1 * ldM]
            << ", ...\n";
  std::cout << "\t    [ " << M[0 * ldM + 1] << ", " << M[1 * ldM + 1]
            << ", ...\n";
  std::cout << "\t    [ "
            << "...";
  std::cout << std::endl;
}

/**
 * Initializes a matrix with random values.
 *
 * @param matrix The matrix to initialize.
 * @param trans The transpose option.
 * @param size The size of the matrix.
 * @param n The number of columns.
 * @param m The number of rows.
 * @param ld The leading dimension of the matrix.
 */
template <typename Ts>
void init_data(Ts &matrix, mklTrans trans, size_t size, std::int64_t n,
               std::int64_t m, std::int64_t ld) {
  using fpt = typename Ts::value_type;

  matrix.resize(size);

  if (trans == mklTrans::nontrans) {
    for (int j = 0; j < n; j++)
      for (int i = 0; i < m; i++)
        matrix.at(i + j * ld) =
            fpt((std::rand()) / fpt(float(RAND_MAX))) - fpt(0.5);
  } else {
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        matrix.at(j + i * ld) =
            fpt((std::rand()) / fpt(float(RAND_MAX))) - fpt(0.5);
  }
}
