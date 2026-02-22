# Численное интегрирование методом трапеций

* Студент: Дергунов Сергей Антонович
* Группа: 3823Б1ПР4
* Вариант: 20  
* Технологии: SEQ, MPI

---

## 1. Введение

Численное интегрирование является одним из ключевых инструментов вычислительной математики
и широко применяется в инженерных и научных расчетах,
когда аналитическое решение интеграла невозможно получить или оно слишком сложное.
Методы численного интегрирования позволяют получать приближенные значения интегралов с заданной точностью.

Метод трапеций — один из базовых способов численного интегрирования.
Он аппроксимирует область под графиком функции трапециями, что позволяет легко реализовать метод
и эффективно использовать параллельные вычисления, так как каждое разбиение интеграла может обрабатываться независимо.

Цель данной работы — реализовать метод трапеций в последовательном (SEQ) и параллельном (MPI) вариантах,
провести тестирование точности, сравнение с аналитическими значениями и оценку производительности параллельной версии.

---

## 2. Постановка задачи

Вычислить интеграл:

I = ∫[a..b] f(x) dx

на сетке из n равных интервалов.

Шаг сетки:

h = (b - a) / n .

Метод трапеций аппроксимирует интеграл как сумму площадей трапеций:

I ≈ h *( (f(a) + f(b))/2 + Σ f(a + k*h) ) for k = 1..n-1 .

В MPI-версии каждая часть суммы распределяется между процессами:

I ≈ Σ ((f(x_i) + f(x_{i+1}))/2) *h ,
где x_i = a + i* h.

### Формат входных данных

```cpp
struct InputParams {
  double a;
  double b;
  int n;
  int func_id; // 0: x, 1: x^2, 2: sin(x)
};
```

Поддерживаемые функции:

* f(x) = x
* f(x) = x²
* f(x) = sin(x)

### Результат

Одно число типа double, приближённое значение интеграла.

---

## 3. Последовательная реализация (SEQ)

Алгоритм SEQ:

1. Вычислить шаг h.
2. Начать сумму с (f(a) + f(b))/2.
3. Просуммировать все значения f(a + k*h) для k = 1..n-1.
4. Умножить результат на h.

Сложность метода:

* Время: O(n).
* Память: O(1).

Последовательная реализация удобна для проверки корректности,
но при больших n её производительность ограничена скоростью одного процессора.

---

## 4. Параллельная реализация (MPI)

### Распределение нагрузки

Чтобы обеспечить равномерное распределение вычислений, диапазон интегрирования делится между процессами:

```cpp
int base  = n / size;
int extra = n % size;

int local_n = base + (rank < extra ? 1 : 0);
int start_i = rank * base + std::min(rank, extra);
int end_i   = start_i + local_n;
```

### Локальные вычисления

Каждый процесс вычисляет сумму трапеций на своей части разбиения.

### Сбор результата

* MPI_Reduce собирает частичные суммы в процессе 0
* MPI_Bcast рассылает итог всем процессам
* Все процессы получают одинаковый результат через GetOutput()

---

## 5. Программная реализация

### 5.1. Архитектура

MPI-версия оформлена в виде класса DergynovSTrapezoidMethodMPI, наследуемого от базового ppc::task::Task.

Этапы работы:

1. Validation — проверка корректности входных данных
2. PreProcessing — подготовка состояния перед вычислениями
3. Run — выполнение интегрирования по MPI
4. PostProcessing — финальная обработка результата

SEQ-версия аналогична, за исключением отсутствия MPI.

---

### 5.2. Структура классов

```cpp
namespace dergynov_s_trapezoid_integration {

class DergynovSTrapezoidIntegrationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit DergynovSTrapezoidIntegrationMPI(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dergynov_s_trapezoid_integration
```

---

### 5.3. Алгоритмы

#### 5.3.1. Конструктор

```cpp
DergynovSTrapezoidIntegrationMPI::DergynovSTrapezoidIntegrationMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}
```

---

### 5.3.2. Валидация

```cpp
bool DergynovSTrapezoidIntegrationMPI::ValidationImpl() {
  const auto& in = GetInput();
  return (in.n > 0) && (in.a < in.b);
}
```

---

### 5.3.3. Предобработка

```cpp
bool DergynovSTrapezoidIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}
```

---

### 5.3.4. Основные вычисления (MPI-версия)

```cpp
bool DergynovSTrapezoidIntegrationMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType in = (rank == 0) ? GetInput() : InType{};
  MPI_Bcast(&in, sizeof(InType), MPI_BYTE, 0, MPI_COMM_WORLD);
  GetInput() = in;

  const double a = in.a;
  const double b = in.b;
  const int n = in.n;

  if (n <= 0 || !(a < b)) {
    if (rank == 0) GetOutput() = 0.0;
    return true;
  }

  int base = n / size;
  int rem = n % size;
  int local_n = base + (rank < rem ? 1 : 0);
  int start = rank * base + std::min(rank, rem);
  int end = start + local_n;

  const double h = (b - a) / static_cast<double>(n);

  double local_sum = 0.0;
  for (int i = start; i < end; ++i) {
    double x1 = a + h * i;
    double x2 = a + h * (i + 1);
    local_sum += 0.5 * (Function(x1, in.func_id) + Function(x2, in.func_id)) * h;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}
```

---

### 5.3.5. Постобработка

```cpp
bool DergunovSTrapezoidMethodMPI::PostProcessingImpl() {
  return true;
}
```

---

## 6. Экспериментальная часть

### Аппаратная конфигурация

* Docker-контейнер (Linux)
* GCC
* OpenMPI
* Режим сборки: Release

### Команды запуска

* mpirun --allow-run-as-root -n 1 ./build/bin/ppc_perf_tests
* mpirun --allow-run-as-root -n 2 ./build/bin/ppc_perf_tests
* mpirun --allow-run-as-root -n 4 ./build/bin/ppc_perf_tests

### Параметры эксперимента

* a = 0.0
* b = 100.0
* n = 10 000 000
* f(x) = x

---

## 7. Результаты

### 7.1 Корректность

* Функциональные тесты SEQ и MPI пройдены.
* Сравнение с аналитическим интегралом и между SEQ и MPI показало совпадение с точностью 1e-3.
* Эффективность параллельного метода напрямую зависит от числа процессов и размера задачи.

---

### 7.2 Производительность

| Режим | Процессы | Время, мс | Ускорение | Эффективность |
| ----: | -------: | --------: | --------: | ------------: |
| SEQ   | 1        | 15        | 1.00      | —             |
| MPI   | 1        | 20        | 0.75      | 75%           |
| MPI   | 2        | 8         | 1.88      | 94%           |
| MPI   | 4        | 83        | 0.18      | 4.5%          |

Комментарий:

* При 2 процессах наблюдается хорошее ускорение.
* При 4 процессах накладные расходы MPI начинают преобладать из-за особенностей запуска в контейнере.
* Эффективность параллельного метода напрямую зависит от числа процессов и размера задачи.

---

## 8. Заключение

* Реализованы последовательная и MPI-версии метода трапеций.
* Проверена корректность работы на нескольких функциях.
* MPI-версия даёт ускорение при умеренном числе процессов.
* Слишком большое число процессов для малой задачи снижает эффективность.

Дальнейшие улучшения:

* Увеличение размера задачи для лучшей параллельной производительности.
* Тестирование на реальном кластере с большим числом узлов.
* Сравнение с более точными методами интегрирования, например, методом Симпсона или адаптивным методом.
* Добавление визуализации разбиений и площади под кривой для наглядного понимания работы метода.

---

## 9. Литература

1. Gropp W., Lusk E., Skjellum A. Using MPI
2. MPI Standard Documentation
3. Документация GoogleTest
4. Сысоев А. В. Лекции по параллельному программированию
