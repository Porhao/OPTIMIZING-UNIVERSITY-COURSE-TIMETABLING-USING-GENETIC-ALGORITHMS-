import random
import matplotlib.pyplot as plt

# University Club Event Scheduling Problem (UCESP)
class UCESP:
    def __init__(self, num_events, num_clubs, num_venues, num_timeslots):

        self.num_events = num_events
        self.num_clubs = num_clubs
        self.num_venues = num_venues
        self.num_timeslots = num_timeslots

        # Example data
        self.clubs = [f"Club_{i}" for i in range(num_clubs)]
        self.venues = [f"Venue_{i}" for i in range(num_venues)]
        self.events = [f"Event_{i}" for i in range(num_events)]
        self.event_club_map = {event: random.choice(self.clubs) for event in self.events}
        self.venue_capacity = {venue: random.randint(50, 200) for venue in self.venues}
        self.event_expected_attendance = {event: random.randint(30, 150) for event in self.events}

    def generate_random_schedule(self):
        schedule = []
        for event in self.events:
            venue = random.choice(self.venues)
            timeslot = random.choice(range(self.num_timeslots))
            schedule.append((event, venue, timeslot))
        return schedule


# Fitness Function
def fitness(schedule, ucesp):
    hard_constraints_violations = 0
    soft_constraints_violations = 0

    # Hard Constraint: No overlapping events in the same venue at the same time
    venue_timeslot_map = {}
    for event, venue, timeslot in schedule:
        if (venue, timeslot) in venue_timeslot_map:
            hard_constraints_violations += 1
        else:
            venue_timeslot_map[(venue, timeslot)] = event

    # Soft Constraint: Venue capacity utilization
    for event, venue, _ in schedule:
        capacity = ucesp.venue_capacity[venue]
        attendance = ucesp.event_expected_attendance[event]
        if attendance > capacity:
            soft_constraints_violations += (attendance - capacity)
        else:
            soft_constraints_violations += (capacity - attendance)

    return 1 / (1 + hard_constraints_violations + soft_constraints_violations)


# Genetic Algorithm
def genetic_algorithm(ucesp, population_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.05):
    population = [ucesp.generate_random_schedule() for _ in range(population_size)]
    best_solution = None
    best_fitness = float('-inf')
    fitness_over_time = []  # Track fitness values over generations

    for generation in range(generations):
        new_population = []
        
        for _ in range(population_size // 2):
            # Tournament Selection
            tournament = random.sample(population, 3)
            parent1 = max(tournament, key=lambda x: fitness(x, ucesp))
            tournament = random.sample(population, 3)
            parent2 = max(tournament, key=lambda x: fitness(x, ucesp))
            
            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, len(parent1) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, len(child1) - 1)
                child1[mutation_point] = ucesp.generate_random_schedule()[mutation_point]
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, len(child2) - 1)
                child2[mutation_point] = ucesp.generate_random_schedule()[mutation_point]
            
            new_population.extend([child1, child2])

        # Update population
        population = new_population

        # Check for the best solution
        for individual in population:
            individual_fitness = fitness(individual, ucesp)
            if individual_fitness > best_fitness:
                best_solution = individual
                best_fitness = individual_fitness

        # Track fitness value for plotting
        fitness_over_time.append(best_fitness)

        print(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
    
    return best_solution, fitness_over_time


# Visualization: Fitness Improvement Over Generations
def plot_fitness(fitness_over_time):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_over_time)), fitness_over_time, marker='o', linestyle='-', color='b')
    plt.title("Fitness Improvement Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Visualization: Gantt Chart of the Schedule
def visualize_schedule(schedule, ucesp):
    """
    Visualizes the event scheduling as a Gantt chart.
    """
    # Create a mapping of venues to Y-axis indices
    venue_indices = {venue: i for i, venue in enumerate(ucesp.venues)}
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add each event to the Gantt chart
    for event, venue, timeslot in schedule:
        club = ucesp.event_club_map[event]
        y = venue_indices[venue]
        ax.broken_barh([(timeslot, 1)], (y - 0.4, 0.8), facecolors='tab:blue', edgecolor='black', label=club)

        # Annotate each event
        ax.text(timeslot + 0.5, y, f"{event} ({club})", color="white", ha="center", va="center", fontsize=8)

    # Add labels, grid, and ticks
    ax.set_yticks(range(len(ucesp.venues)))
    ax.set_yticklabels(ucesp.venues)
    ax.set_xticks(range(ucesp.num_timeslots))
    ax.set_xticklabels([f"Timeslot {i}" for i in range(ucesp.num_timeslots)])
    ax.set_xlabel("Timeslots")
    ax.set_ylabel("Venues")
    ax.set_title("Event Schedule Visualization")
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Initialize UCESP with parameters: num_events, num_clubs, num_venues, num_timeslots
    ucesp = UCESP(num_events=10, num_clubs=5, num_venues=3, num_timeslots=6)

    # Run the genetic algorithm
    best_schedule, fitness_over_time = genetic_algorithm(
        ucesp, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1
    )

    # Output the best schedule found
    print("\nBest Schedule:")
    for event, venue, timeslot in best_schedule:
        print(f"{event} -> {venue} at Timeslot {timeslot}")

    # Visualize fitness improvement over generations
    plot_fitness(fitness_over_time)

    # Visualize the best schedule as a Gantt chart
    visualize_schedule(best_schedule, ucesp)
