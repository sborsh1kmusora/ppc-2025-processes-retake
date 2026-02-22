Поиск минимального значения в векторе

Студентка: Шкрылёва С.А., группа 3823Б1ПР1

Технология: MPI

Вариант: 4

1. Введение

Параллельные вычисления являются важным направлением современного программирования,
позволяющим значительно ускорить обработку больших объёмов данных. В данной работе
рассматривается задача поиска минимального элемента в векторе с использованием
технологии MPI (Message Passing Interface) для распределения вычислений между
несколькими процессами.

Цель работы — реализовать корректный и масштабируемый параллельный алгоритм поиска
минимума и провести анализ его производительности.

2. Постановка задачи

Разработать параллельный алгоритм нахождения минимального значения в векторе целых чисел.

Требования:

корректная работа при различных размерах входных данных;

поддержка произвольного количества MPI-процессов;

равномерное распределение нагрузки;

сравнение с последовательной версией.

3. Последовательный алгоритм

```cpp
int min_val = GetInput()[0];
for (size_t i = 1; i < GetInput().size(); i++) {
  if (GetInput()[i] < min_val) {
    min_val = GetInput()[i];
  }
}
```

Алгоритм выполняет последовательный обход массива.

Временная сложность:
O(n)

Преимущества:

Простота реализации

Минимальные накладные расходы

4. Схема распараллеливания
4.1 Распределение данных

Используется блочное распределение:

Вектор делится на приблизительно равные части.

Если размер не кратен числу процессов — первые процессы получают на один элемент больше.

Размеры вычисляются динамически.

```cpp
int base_size = total_size / world_size;
int extra_items = total_size % world_size;

for (int i = 0; i < world_size; ++i) {
    sendcounts[i] = base_size + (i < extra_items ? 1 : 0);
}
```

4.2 Коммуникационная схема

MPI_Bcast
Процесс 0 рассылает размер входного вектора всем процессам.

MPI_Scatterv
Распределение данных с учётом неравномерных блоков.

Локальные вычисления
Каждый процесс находит минимум в своём блоке.

MPI_Allreduce (MPI_MIN)
Глобальная редукция локальных минимумов.

5. Детали реализации
Структура проекта
seq/      — последовательная версия (ops_seq.hpp/cpp)
mpi/      — параллельная версия (ops_mpi.hpp/cpp)
tests/    — функциональные и performance тесты
Основные методы
ValidationImpl()

Проверка непустого входного вектора

Проверка корректной инициализации выходного значения

PreProcessingImpl()

Инициализация MPI

Установка начального значения результата (INT_MAX)

RunImpl()

Основные шаги:

Получение rank и world_size

Broadcast общего размера данных

Расчёт sendcounts и displacements

MPI_Scatterv

Локальный поиск минимума

MPI_Allreduce с операцией MPI_MIN

PostProcessingImpl()

Завершение MPI

Финальная проверка результата

Обработка краевых случаев
Ситуация Поведение
Пустой вектор Возврат false
1 процесс Работает как sequential
Процесс без данных Использует INT_MAX
Потребление памяти

Каждый процесс хранит только свой локальный блок

Процесс 0 хранит полный вектор

Дополнительная память: sendcounts, displacements

6. Экспериментальная установка
Аппаратное обеспечение

CPU: AMD Ryzen 5 4600H (6 ядер, 12 потоков)

RAM: 16 GB

ОС: Windows 10

Инструменты

CMake

MSVC 14.37.32822

Microsoft MPI 10.1.12498.52

Тип сборки: Release

Visual Studio Code

Параметры тестирования

Размер вектора: 100 000 000 элементов

Диапазон значений: [-1000, 1000]

Процессы: 1, 2, 4, 8

7. Результаты экспериментов
7.1 Корректность

Все функциональные тесты пройдены

Результаты MPI и последовательной версии совпадают

7.2 Производительность
Процессы Время (с) Ускорение Эффективность
1 (seq) 0.082347 1.00 100%
2 0.203896 0.40 20%
4 0.184119 0.45 11%
8 0.154157 0.53 7%
Анализ

Параллельная версия медленнее последовательной.

Основная причина — накладные расходы MPI.

Наблюдается небольшое улучшение от 2 к 8 процессам.

Задача слишком простая для эффективной параллелизации.

8. Выводы

Алгоритм реализован корректно.

Все тесты успешно пройдены.

Для задачи линейного поиска минимума MPI не даёт ускорения.

Накладные расходы коммуникаций превышают выигрыш от распараллеливания.

MPI становится эффективным при более вычислительно сложных задачах.

Работа демонстрирует корректную реализацию параллельного алгоритма и понимание принципов распределённых вычислений.

9. Источники

MPI Standard — <https://www.mpi-forum.org/docs/>

MPICH Documentation — <https://www.mpich.org/documentation/guides/>

Microsoft MPI — <https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi>

OpenMPI Documentation — <https://www.open-mpi.org/docs/>

Приложение

```cpp
bool ShkrylevaSVecMinValMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  const std::vector<int>* input_data_ptr = nullptr;

  if (world_rank == 0) {
    input_data_ptr = &GetInput();
    total_size = static_cast<int>(input_data_ptr->size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    return false;
  }

  int base_size = total_size / world_size;
  int extra_items = total_size % world_size;

  std::vector<int> sendcounts(world_size);
  std::vector<int> displacements(world_size);

  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    sendcounts[i] = base_size + (i < extra_items ? 1 : 0);
    displacements[i] = offset;
    offset += sendcounts[i];
  }

  std::vector<int> local_data(sendcounts[world_rank]);

  MPI_Scatterv(
      (world_rank == 0) ? input_data_ptr->data() : nullptr,
      sendcounts.data(),
      displacements.data(),
      MPI_INT,
      local_data.data(),
      sendcounts[world_rank],
      MPI_INT,
      0,
      MPI_COMM_WORLD);

  int local_min = INT_MAX;
  for (int value : local_data) {
    local_min = std::min(local_min, value);
  }

  int total_min = INT_MAX;
  MPI_Allreduce(&local_min, &total_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = total_min;

  return true;
}
```
