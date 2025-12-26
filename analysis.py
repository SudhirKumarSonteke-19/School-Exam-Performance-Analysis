import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub


def load_data():
    path = kagglehub.dataset_download(
        "spscientist/students-performance-in-exams"
    )
    return pd.read_csv(path + "/StudentsPerformance.csv")


def clean_data(df):
    df.columns = df.columns.str.replace(" ", "_")
    return df


def analyze_performance(df):
    subjects = ["math_score", "reading_score", "writing_score"]
    avg_scores = df[subjects].mean()
    return subjects, avg_scores


def show_insights(avg_scores):
    print("Average Subject-wise Scores:\n")
    print(avg_scores)
    print("\nStrongest Subject:", avg_scores.idxmax())
    print("Weakest Subject:", avg_scores.idxmin())


def visualize(df, subjects, avg_scores):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    avg_scores.plot(kind="bar")
    plt.title("Average Subject-wise Scores")
    plt.ylabel("Marks")

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df[subjects])
    plt.title("Marks Distribution")

    plt.subplot(1, 3, 3)
    plt.pie(
        avg_scores,
        labels=subjects,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Subject-wise Performance Share")

    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    df = clean_data(df)
    subjects, avg_scores = analyze_performance(df)
    show_insights(avg_scores)
    visualize(df, subjects, avg_scores)


if __name__ == "__main__":
    main()
