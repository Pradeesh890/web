import streamlit as st

st.title("Simple calculator")
shell_script = """
#!/bin/sh

echo "Enter the two numbers: "
read a b

echo "5.Addition"
echo "6.Subtraction"
echo "7.Multiplication"
echo "8.Division"
echo "Enter the option: "
read option

case $option in
1)
    c=$(expr $a + $b)
    echo "$a + $b = $c"
    ;;
2)
    c=$(expr $a - $b)
    echo "$a - $b = $c"
    ;;
3)
    c=$(expr $a \* $b)
    echo "$a * $b = $c"
    ;;
4)
    c=$(expr $a / $b)
    echo "$a / $b = $c"
    ;;
*)
    echo "Invalid Option"
    ;;
esac
"""

st.code(shell_script, language='bash')

st.title("fibonacci series")
fibo="""echo "How many number of terms to be generated?" read n
x=0 y=1 i=2
echo "Fibonacci Series up to $n terms:" echo "$x"
echo "$y"
while [ $i -lt $n ] do
i=`expr $i + 1 ` z=`expr $x + $y ` echo "$z"
x=$y y=$z done """
st.code(fibo,language='bash')


st.title("System Call - Wait System Call")
syscall="""#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == -1) {
        perror("Process creation unsuccessful ");
        exit(EXIT_FAILURE);
    } else if (pid > 0) {
        wait(NULL);
        printf("Parent starts ");

        for (int i = 0; i <= 10; i += 2) {
            printf("%d ", i);
        }
        printf(" ");

        printf("Parent ends ");
    } else {
        printf("Child starts ");

        for (int i = 1; i <= 10; i += 2) {
            printf("%d ", i);
        }
        printf(" ");

        printf("Child ends ");

        exit(EXIT_SUCCESS);
    }

    return 0;
}
"""
st.code(syscall,language='cpp')


st.title("CPU scheduling FCFS")
fcfs="""#include <stdio.h>

struct Process {
    char name;
    int burst_time;
    int arrival_order;
    int waiting_time;
    int turnaround_time;
};

void calculateTimes(struct Process *p1, struct Process *p2) {
    p1->waiting_time = 0;
    p1->turnaround_time = p1->burst_time;
    p2->waiting_time = p1->burst_time;
    p2->turnaround_time = p2->waiting_time + p2->burst_time;
}

int main() {
    struct Process p1, p2;

    printf("Enter the name of process 1: ");
    scanf(" %c", &p1.name);
    printf("Enter the burst time of process 1: ");
    scanf("%d", &p1.burst_time);
    printf("Enter the arrival order of process 1: ");
    scanf("%d", &p1.arrival_order);

    printf("Enter the name of process 2: ");
    scanf(" %c", &p2.name);
    printf("Enter the burst time of process 2: ");
    scanf("%d", &p2.burst_time);
    printf("Enter the arrival order of process 2: ");
    scanf("%d", &p2.arrival_order);

    if (p1.arrival_order > p2.arrival_order) {
        struct Process temp = p1;
        p1 = p2;
        p2 = temp;
    }

    calculateTimes(&p1, &p2);

    printf("Process %c:", p1.name);
    printf("Waiting time: %d", p1.waiting_time);
    printf("Turnaround time: %d", p1.turnaround_time);

    printf("Process %c:", p2.name);
    printf("Waiting time: %d", p2.waiting_time);
    printf("Turnaround time: %d", p2.turnaround_time);

    return 0;
}
"""
st.code(fcfs,language='cpp')








st.title("SJF CPU scheduling")
sjf="""#include <stdio.h>
#include <string.h>

struct Process {
    char name[10];
    int burst_time;
    int arrival_order;
    int waiting_time;
    int turnaround_time;
};

void sort_by_burst_time(struct Process *p, int n) {
    struct Process temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (p[j].burst_time > p[j + 1].burst_time) {
                temp = p[j];
                p[j] = p[j + 1];
                p[j + 1] = temp;
            }
        }
    }
}

void calculate_times(struct Process *p, int n) {
    p[0].waiting_time = 0;
    p[0].turnaround_time = p[0].burst_time;
    
    for (int i = 1; i < n; i++) {
        p[i].waiting_time = p[i - 1].turnaround_time;
        p[i].turnaround_time = p[i].waiting_time + p[i].burst_time;
    }
}

int main() {
    struct Process p[2];
    float avg_waiting_time, avg_turnaround_time;
    
    for (int i = 0; i < 2; i++) {
        printf("Enter the name of process %d: ", i + 1);
        scanf("%s", p[i].name);
        printf("Enter the burst time of process %d: ", i + 1);
        scanf("%d", &p[i].burst_time);
        printf("Enter the arrival order of process %d: ", i + 1);
        scanf("%d", &p[i].arrival_order);
    }

    sort_by_burst_time(p, 2);
    calculate_times(p, 2);

    printf("Process Details");
    printf("Process NameCPUBurstTimeArrivalOrderWaitingTimeTurnaroundTime");

    for (int i = 0; i < 2; i++) {
        printf("%s%d%d%d%d", p[i].name, p[i].burst_time, p[i].arrival_order, p[i].waiting_time, p[i].turnaround_time);
    }

    avg_waiting_time = (p[0].waiting_time + p[1].waiting_time) / 2.0;
    avg_turnaround_time = (p[0].turnaround_time + p[1].turnaround_time) / 2.0;

    printf("Average waiting time is %.2f", avg_waiting_time);
    printf("Average turnaround time is %.2f", avg_turnaround_time);

    return 0;
}
"""

st.code(sjf,language='cpp')





st.title("Priority Scheduling")

priority="""#include <stdio.h>
#include <stdlib.h>

struct Process {
    char name;
    int burst_time;
    int priority;
    int waiting_time;
    int turnaround_time;
};

void swap(struct Process *xp, struct Process *yp) {
    struct Process temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void sortProcesses(struct Process *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j].priority > arr[j + 1].priority) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

void calculateTimes(struct Process *p, int n) {
    p[0].waiting_time = 0;
    p[0].turnaround_time = p[0].burst_time;

    for (int i = 1; i < n; i++) {
        p[i].waiting_time = p[i - 1].turnaround_time;
        p[i].turnaround_time = p[i].waiting_time + p[i].burst_time;
    }
}

int main() {
    struct Process processes[2];
    float avg_waiting_time, avg_turnaround_time;

    // Input
    for (int i = 0; i < 2; i++) {
        printf("Enter the name of process %d: ", i + 1);
        scanf(" %c", &processes[i].name);
        printf("Enter the burst time of process %d: ", i + 1);
        scanf("%d", &processes[i].burst_time);
        printf("Enter the priority of process %d: ", i + 1);
        scanf("%d", &processes[i].priority);
    }

    // Sort processes based on priority
    sortProcesses(processes, 2);

    // Calculate times
    calculateTimes(processes, 2);

    // Display process details
    printf("Process Details");
    printf("Process Name    Burst Time    Priority    Waiting Time    turnaround Time");
    for (int i = 0; i < 2; i++) {
        printf("%c        %d        %d        %d        %d", processes[i].name, processes[i].burst_time,
               processes[i].priority, processes[i].waiting_time, processes[i].turnaround_time);
    }

    return 0;
}
"""
st.code(priority,language='cpp')



st.title("Round Robin")
roundrob="""#include <stdio.h>
#define MAX_PROCESSES 2

struct Process {
    char name;
    int burst_time;
    int remaining_time;
    int waiting_time;
    int turnaround_time;
};

void roundRobin(struct Process *processes, int n, int time_slice) {
    int total_time = 0;
    int completed = 0;
    
    // Initialize remaining time for all processes
    for (int i = 0; i < n; i++) {
        processes[i].remaining_time = processes[i].burst_time;
    }
    
    // Round Robin scheduling
    while (completed < n) {
        for (int i = 0; i < n; i++) {
            if (processes[i].remaining_time > 0) {
                if (processes[i].remaining_time > time_slice) {
                    total_time += time_slice;
                    processes[i].remaining_time -= time_slice;
                } else {
                    total_time += processes[i].remaining_time;
                    processes[i].waiting_time = total_time - processes[i].burst_time;
                    processes[i].turnaround_time = total_time;
                    processes[i].remaining_time = 0;
                    completed++;
                }
            }
        }
    }
}

int main() {
    struct Process processes[MAX_PROCESSES];
    int time_slice;

    // Input
    for (int i = 0; i < MAX_PROCESSES; i++) {
        printf("Enter the name of process %d: ", i + 1);
        scanf(" %c", &processes[i].name);
        printf("Enter the burst time of process %d: ", i + 1);
        scanf("%d", &processes[i].burst_time);
    }

    printf("Enter the time slice: ");
    scanf("%d", &time_slice);

    // Apply Round Robin scheduling
    roundRobin(processes, MAX_PROCESSES, time_slice);

    // Display process details
    printf("Process Details");
    printf("Process Name  CPU Burst Time  Waiting Time  Turnaround Time  ");
    for (int i = 0; i < MAX_PROCESSES; i++) {
        printf("%c %d %d %d", processes[i].name, processes[i].burst_time,
               processes[i].waiting_time, processes[i].turnaround_time);
    }

    return 0;
}
"""
st.code(roundrob,language='cpp')


st.title("Bankers algorithm or deadlock avoidance")
bank="""#include <stdio.h>
#include <stdbool.h>

#define MAX_PROCESSES 10
#define MAX_RESOURCES 10

int processes, resources;
int available[MAX_RESOURCES];
int maximum[MAX_PROCESSES][MAX_RESOURCES];
int allocation[MAX_PROCESSES][MAX_RESOURCES];
int need[MAX_PROCESSES][MAX_RESOURCES];

void calculateNeed() {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need[i][j] = maximum[i][j] - allocation[i][j];
        }
    }
}

bool isSafe(int processOrder[], bool finish[]) {
    int work[MAX_RESOURCES];
    bool safeSequence = true;

    // Initialize work array with available resources
    for (int i = 0; i < resources; i++) {
        work[i] = available[i];
    }

    // Initialize finish array
    for (int i = 0; i < processes; i++) {
        finish[i] = false;
    }

    // Find a safe sequence
    int count = 0;
    while (count < processes) {
        bool found = false;
        for (int p = 0; p < processes; p++) {
            if (!finish[p]) {
                bool canExecute = true;
                for (int r = 0; r < resources; r++) {
                    if (need[p][r] > work[r]) {
                        canExecute = false;
                        break;
                    }
                }
                if (canExecute) {
                    // Execute process p
                    for (int r = 0; r < resources; r++) {
                        work[r] += allocation[p][r];
                    }
                    processOrder[count++] = p;
                    finish[p] = true;
                    found = true;
                }
            }
        }
        if (!found) {
            safeSequence = false;
            break;
        }
    }

    return safeSequence;
}

void printSequence(int processOrder[]) {
    printf("Safe sequence:");
    for (int i = 0; i < processes; i++) {
        printf("P%d ", processOrder[i]);
        printf("Availability");
        for (int j = 0; j < resources; j++) {
            printf(" R%d %d", j + 1, available[j]);
        }
        printf(" ");
    }
}

int main() {
    printf("Enter number of processes: ");
    scanf("%d", &processes);

    printf("Enter number of resources: ");
    scanf("%d", &resources);

    printf("Enter available resources:");
    for (int i = 0; i < resources; i++) {
        scanf("%d", &available[i]);
    }

    printf("Enter allocation matrix:");
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            scanf("%d", &allocation[i][j]);
        }
    }

    printf("Enter maximum matrix:");
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            scanf("%d", &maximum[i][j]);
        }
    }

    calculateNeed();

    int processOrder[MAX_PROCESSES];
    bool finish[MAX_PROCESSES];

    if (isSafe(processOrder, finish)) {
        printSequence(processOrder);
    } else {
        printf("System is in deadlock");
    }

    return 0;
}
"""
st.code(bank,language='cpp')

st.title("squential file allocation")
sequ="""#include <stdio.h>

int main() {
    int f[50] = {0}; // Array to represent free/allocated blocks
    int st, len, c;
    
    do {
        printf("Enter the starting block and length of the file: ");
        scanf("%d %d", &st, &len);
        
        // Check if the blocks are free and allocate them
        int allocated = 1; // Assume allocation is successful unless proven otherwise
        for (int j = st; j < st + len; j++) {
            if (f[j] == 0) {
                f[j] = 1; // Allocate block
                printf("%d -> %d", j, f[j]); // Print allocated block
            } else {
                printf("Block %d is already allocated.", j);
                allocated = 0; // Allocation failed
                break;
            }
        }
        
        // If all blocks were successfully allocated
        if (allocated) {
            printf("The file is allocated to disk.");
        }
        
        printf("Do you want to enter more files? (1 for Yes / 0 for No): ");
        scanf("%d", &c);
    } while (c == 1);
    
    printf("Exiting program.");
    
    return 0;
}
"""
st.code(sequ,language='cpp')

st.title("Linked file allocation")
link="""#include <stdio.h>

int main() {
    int f[50] = {0}; // Array to represent free/allocated blocks
    int p, st, len, c;
    
    // Input: Blocks already allocated
    printf("Enter how many blocks are already allocated: ");
    scanf("%d", &p);
    
    printf("Enter the blocks numbers that are already allocated: ");
    for (int i = 0; i < p; i++) {
        int a;
        scanf("%d", &a);
        f[a] = 1; // Mark allocated blocks
    }
    
    // File allocation process
    do {
        printf("Enter the starting index block and length: ");
        scanf("%d %d", &st, &len);
        
        int k = len;
        for (int j = st; j < st + k; j++) {
            if (f[j] == 0) {
                f[j] = 1; // Allocate block
                printf("%d -> %d", j, f[j]); // Print allocated block
            } else {
                printf("%d -> File is already allocated", j);
                k++; // Increase length to cover the already allocated blocks
            }
        }
        
        printf("Do you want to enter one more file? (1 for Yes / 0 for No): ");
        scanf("%d", &c);
    } while (c == 1);
    
    printf("Exiting program.\n");
    
    return 0;
}
"""

st.code(link,language='cpp')

st.title("indexed file allocation")
index="""#include <stdio.h>

int main() {
    int f[50] = {0}; // Array to represent free/allocated blocks
    int inde[50]; // Array to store indices of files
    int p, n, c;

    // Initialize f array to 0 (all blocks free)
    for (int i = 0; i < 50; i++) {
        f[i] = 0;
    }

    // Main loop to handle file allocation
    do {
        printf("Enter index block: ");
        scanf("%d", &p);

        // Check if index block is already allocated
        if (f[p] == 1) {
            printf("Block already allocated.");
            continue;
        }

        // Allocate index block
        f[p] = 1;

        printf("Enter number of files on index: ");
        scanf("%d", &n);

        // Input files for the index block
        printf("Enter %d file indices: ", n);
        for (int i = 0; i < n; i++) {
            scanf("%d", &inde[i]);
            
            // Check if file block is already allocated
            if (f[inde[i]] == 1) {
                printf("Block %d already allocated.", inde[i]);
                f[p] = 0; // Deallocate index block
                break;
            }
            
            // Allocate file block
            f[inde[i]] = 1;
        }

        // If all files are successfully allocated
        if (f[p] == 1) {
            printf("Allocated.");
            printf("File indexed:");
            for (int k = 0; k < n; k++) {
                printf("%d -> %d:%d", p, inde[k], f[inde[k]]);
            }
        }

        printf("Enter 1 to enter more files and 0 to exit: ");
        scanf("%d", &c);
    } while (c == 1);

    printf("Exiting program.");

    return 0;
}
"""
st.code(index,language='cpp')


st.title("PROCESS SYNCHRONIZATION – PRODUCER CONSUMER PROBLEM")
propro="""#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h> // Include for sleep function

#define BUFFER_SIZE 5

sem_t empty_slots, full_slots;
pthread_mutex_t mutex;

int buffer[BUFFER_SIZE];
int in = 0, out = 0;

void *producer(void *arg) {
    int item = 1; // Item to produce

    while (1) {
        // Wait for empty slot
        sem_wait(&empty_slots);

        // Acquire mutex
        pthread_mutex_lock(&mutex);

        // Produce item and add to buffer
        buffer[in] = item;
        printf("Produced item: %d\n", item);
        in = (in + 1) % BUFFER_SIZE;

        // Release mutex
        pthread_mutex_unlock(&mutex);

        // Signal full slot
        sem_post(&full_slots);

        item++;
        sleep(1); // Simulate some processing time
    }
    pthread_exit(NULL);
}

void *consumer(void *arg) {
    while (1) {
        // Wait for full slot
        sem_wait(&full_slots);

        // Acquire mutex
        pthread_mutex_lock(&mutex);

        // Consume item from buffer
        int item = buffer[out];
        printf("Consumed item: %d\n", item);
        out = (out + 1) % BUFFER_SIZE;

        // Release mutex
        pthread_mutex_unlock(&mutex);

        // Signal empty slot
        sem_post(&empty_slots);

        sleep(2); // Simulate some processing time
    }
    pthread_exit(NULL);
}

int main() {
    pthread_t producer_thread, consumer_thread;

    // Initialize semaphores and mutex
    sem_init(&empty_slots, 0, BUFFER_SIZE);
    sem_init(&full_slots, 0, 0);
    pthread_mutex_init(&mutex, NULL);

    // Create producer and consumer threads
    pthread_create(&producer_thread, NULL, producer, NULL);
    pthread_create(&consumer_thread, NULL, consumer, NULL);

    // Wait for threads to finish (which they won't)
    pthread_join(producer_thread, NULL);
    pthread_join(consumer_thread, NULL);

    // Clean up
    sem_destroy(&empty_slots);
    sem_destroy(&full_slots);
    pthread_mutex_destroy(&mutex);

    return 0;
}
"""
st.code(propro,language='cpp')


st.title("PROCESS SYNCHRONIZATION – DINING PHILOSOPHER’S PROBLEM ")
prosync="""#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <unistd.h>

#define N 5 // Number of philosophers/chopsticks

#define THINKING 2
#define HUNGRY 1
#define EATING 0

sem_t mutex;
sem_t S[N]; // Semaphores for each philosopher

int state[N]; // State of each philosopher

void *philosopher(void *num);
void take_chopstick(int phnum);
void put_chopstick(int phnum);
void test(int phnum);

void *philosopher(void *num) {
    while (1) {
        int *i = num;
        usleep(1000); // Simulate thinking
        take_chopstick(*i);
        usleep(1000); // Simulate eating
        put_chopstick(*i);
    }
}

void take_chopstick(int phnum) {
    sem_wait(&mutex);
    state[phnum] = HUNGRY;
    printf("Philosopher %d is hungry", phnum + 1);
    test(phnum);
    sem_post(&mutex);
    sem_wait(&S[phnum]);
    usleep(1000); // Simulate grabbing chopstick
}

void put_chopstick(int phnum) {
    sem_wait(&mutex);
    state[phnum] = THINKING;
    printf("Philosopher %d is putting chopsticks down", phnum + 1);
    test(LEFT);
    test(RIGHT);
    sem_post(&mutex);
}

void test(int phnum) {
    if (state[phnum] == HUNGRY && state[LEFT] != EATING && state[RIGHT] != EATING) {
        state[phnum] = EATING;
        printf("Philosopher %d takes chopsticks %d and %d", phnum + 1, LEFT + 1, phnum + 1);
        sem_post(&S[phnum]);
    }
}

int main() {
    pthread_t thread_id[N];
    sem_init(&mutex, 0, 1);

    for (int i = 0; i < N; i++)
        sem_init(&S[i], 0, 0);

    for (int i = 0; i < N; i++) {
        pthread_create(&thread_id[i], NULL, philosopher, &i);
        printf("Philosopher %d is thinking", i + 1);
    }

    for (int i = 0; i < N; i++)
        pthread_join(thread_id[i], NULL);

    for (int i = 0; i < N; i++)
        sem_destroy(&S[i]);

    sem_destroy(&mutex);

    return 0;
}
"""
st.code(prosync,language='cpp')
