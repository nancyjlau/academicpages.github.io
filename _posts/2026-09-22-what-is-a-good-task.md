---
title: "What is a good RL task?"
date: 2026-09-22
permalink: /posts/2026/09/what-is-a-good-task/
excerpt: |
  Benchmarks are made up of tasks, and the quality of your benchmark comes down to the quality of each one.

  A task is a problem you give an agent, where it gets tools and files to work with, along with a way to score its output. There are three parts to…
canonical_url: https://www.hud.ai/research/what-is-a-good-task
original_publication: HUD
original_authors: "Reinis Salaks, Ilya Kuhta, and Nancy Lau"
tags:
  - AI agents
  - Reinforcement learning
  - Evaluation
---

{% include mirror-notice.html %}

![What is a good RL task?]({{ '/images/posts/what-is-a-good-task/task-cover.png' | relative_url }})

Benchmarks are made up of tasks, and the quality of your benchmark comes down to the quality of each one.

A task is a problem you give an agent, where it gets tools and files to work with, along with a way to score its output. There are three parts to a task:

1. The prompt (the question that you ask)
2. The environment (the tools and files provided to the agent)
3. The grader (how you check whether the problem was solved or not)

You don’t always need all three. Given the popular “How many Rs are in strawberry?” or “What year was the Eiffel tower completed?” questions, there’s only a single prompt and answer with no environment needed. But most realistic tasks are much more involved, and a lot of them have more than one valid answer where the resulting state (was a hotel/flight booking made?) needs to be checked rather than a simple static answer.

At HUD, we’ve helped build tasks across coding, browser work, robotics, and cybersecurity. When issues are found in a benchmark, the problems usually lie in the task itself, whether it’s a badly scoped problem, broken grading, or something else entirely. In this blog, we’ll talk about what makes a good task. We define a good task as one that’s realistic, fair, and difficult.

## Why this matters more for training than for evals

You can use a task for evaluation (how well does my agent do?) or for RL training (let’s improve our agent!), but there’s a core difference here.

In evaluation, a broken grader could allow your agent to score well without actually solving the problem (reward hacking). But in training, an agent has thousands of attempts to reward hack again and again, in a training loop that reinforces such behavior. Over time, your agent learns to exploit hacks in your grader instead of learning what you intended in the original task.

When designing a task, you should assume that your agent will eventually try any possible approach to get a passing score, whether it’s a legitimate attempt at solving the problem or exploiting shortcuts. This is why realism, fairness, and difficulty in a hardened task are not just nice-to-haves, a broken task can teach your agent completely different things.

## A running example

Let’s take a look at a real booking request task to break down what these three look like in practice.

> You are acting as a receptionist at a music school and a parent would like to book the first piano lesson for her daughter, Mia, for a full hour with Ms. Nowak. They will not be able to arrive before 15:30 and Mia has to be home for dinner, so the lesson must end by 18:00. It has to take place on a weekday in the week starting on Monday 10 August. Since Ms. Nowak only teaches on Mondays, Tuesdays and Thursdays, you should use the timetable to find out when she is free.

This booking request is the prompt, which we provide to an agent to complete the task. The agent is provided an environment where it has access to information needed to solve the task (the teaching schedule and tooling). When the agent checks the music school schedule, it can find a correct slot that fulfills the request (Thursday 13 August, 16:00–17:00). When the task is done, the grader checks if the final result was correct.

Using [HUD’s SDK](https://github.com/hud-evals/hud-python), that same shape looks like this:

```python
from pathlib import Path

from hud.environment import Environment

ROOT = Path("workspace")
env = Environment(name="booking")
env.workspace(ROOT)

TASK_PROMPT = """You are acting as a receptionist at a music school and a parent would like to book the first piano lesson for her daughter, Mia, for a full hour with Ms. Nowak..."""

@env.template(id="piano_lesson")
async def piano_lesson():
    answer = yield TASK_PROMPT
    yield 1.0 if verify_booking(answer, day="Thursday", start="16:00", end="17:00") else 0.0
```

Read more on the technical details in [Creating an Environment](https://docs.hud.ai/v6/guides/creating-an-environment).

You can check the whole trace from prompt to results [in the original article on HUD](https://www.hud.ai/research/what-is-a-good-task#a-running-example).

A good task needs to get three things right: It needs to be realistic, fair, and hard, in that order, because each is a building block for the next.

## 1. Realistic

Your prompt should sound like something a real person would ask, and the work to complete should be something someone actually does. The booking request we have above is a good example, since this is a task a real life receptionist would get.

Don’t frame it as a benchmark, don’t leak the rubric into the environment, and start from real work when you can: an actual UI, bug, or codebase that’s widely used. Grading should always be done separately, somewhere the agent will never interact with or know about.

A task could be graded correctly and be considered difficult, but if it isn’t a realistic task that would be done in real life, then there’s no point in measuring this activity.

Here below is an example of an unrealistic, bad prompt:

```markdown
# 🎯 Task: Optimal Piano Lesson Slot Identification

## 📋 Overview
Welcome! In this task, you will be evaluated on your ability to perform multi-constraint scheduling analysis. Let's dive in! 🚀

[…]

## ✅ Success Criteria
Your response will be considered successful if it:
- Correctly identifies the day (weight: 40%)
- Correctly identifies the time range (weight: 40%)
- Does not propose invalid alternatives (weight: 20%)

I hope this helps! Let me know if you have any questions. 😊
```

No receptionist in the real world has ever received this message. This prompt announces to the agent that this is clearly an evaluation, and shows the rubric grading weights. Although the agent can still functionally complete what we want it to do, this no longer is a realistic task.

Not only do you need realism in your prompt, this extends to the environment that the agent has access to. If your workspace is a bare container with no real files, no realistic state, and nothing a person would actually expect to find on the job, you’re measuring how an agent behaves in a test harness instead of real work. There should be no signals that this is a test environment.

## 2. Fairness

Your task’s grader needs to reward the correct score according to the results and behavior from the agents. There should be no false positives and no false negatives. So how do you design a good grader?

If your grader relies on keyword matching, only accepts one exact phrase, or scores on content that isn’t asked for by the prompt, then this isn’t a fair grader. You should also account for edge cases where your grader could error out given a correct result. One good sanity check is to imagine giving out this task to 10 competent people. Their answer could all functionally be similar, with different methods of approaching the solution that should all score 100%.

If your grader can’t handle cases like these, then there’s more work you need to do.

A grader also has to handle cases you didn’t account for. For example, we had a task where agents had to fix an issue in an attention layer implementation. Four independent agents produced working implementations, but the grader rejected all of them because each agent stored a repeat count in a local variable, and the grader only checked for lines of code that used that variable.

We ended up having to change the grader to check the functionality of the code itself, making sure that it accepted those solutions while rejecting the original issue in the code.

Another case was where we had an agent investigate air quality sensor incidents, and explain what went wrong in each one. The report that it wrote identified all incidents that occurred and categorized them correctly, and reasoned why they occurred for 37/38 incidents. But, the grader marked down the rewarded score because the rule it was checking for treated any `/` in the result as an unwanted file path, and the result had a citation to `ug/m3` (the unit for pollutant concentration).

This is an example of a check meant to catch hallucinations penalizing correct answers, where the model capability and task itself was not the issue!

There are four things that we should keep in mind when designing a good grader. Take the booking task as an example:

1. Provide enough information in the prompt for the grader to be strict. If your prompt just says “let me know what time might work”, you aren’t able to correctly grade for a specific slot. The prompt needs to allow for enough constraints in the answer so that we have a worthwhile grader to check the answer.
2. No static keyword matching. If the grader is marking any answer containing “Thursday” as correct, there would be an issue if the result is "Thursday is fully booked, I'm afraid. I can offer Wednesday 12 August at 16:00." This result would be completely wrong, but it scores 100% because the keyword Thursday is present.

    ![A wrong answer scores 100% because it mentions Thursday.]({{ '/images/posts/what-is-a-good-task/keyword-match.png' | relative_url }})

    *A wrong answer scores 100% because it mentions Thursday.*

3. Check correctness, not your specific answer. In tasks like investigations, scheduling, analysis or Q&A, answers like “Thursday the 13th from 4pm till 5pm” and “Thursday at 16:00 to 17:00” are the same answer. Your grader needs to accept both answers. In coding tasks, this means fully testing for code functionality compared to the reference solution, not whether they match exactly line by line.

    ![A correct answer scores 0% because it is worded differently.]({{ '/images/posts/what-is-a-good-task/exact-match.png' | relative_url }})

    *A correct answer scores 0% because it is worded differently.*

4. Only grade what the prompt asked for. If your grader checks for things the prompt didn’t ask about, you’ll get unreasonable results. Say your grader splits the score between finding the right time slot, and mentioning the correct teacher (Ms. Nowak) by name. But if the prompt never asks for Ms. Nowak by name, an agent could find an open slot that says “the teacher is free at 16:00” and score a 50%, with no way to know why it lost out on the other portion of the score.

    Conversely, another agent that mentions Ms. Nowak by name but doesn’t complete the task would be awarded an undesired score of 50%.

    ![A wrong answer still earns 50% for mentioning Ms. Nowak by name.]({{ '/images/posts/what-is-a-good-task/partial-credit.png' | relative_url }})

    *A wrong answer still earns 50% for mentioning Ms. Nowak by name.*

Remember that in a training loop, having a bad grader doesn’t just give you wrong scores. It also teaches your agent to optimize for whatever the grader actually rewards, just not in the way you intended it to.

## 3. Difficulty

Your task is hard when strong frontier models struggle on it. We often have a passing bar of 50% or less, but this only means anything if you take into account which model you ran, what harness was driving the agent, how you scored the task, and how many times you ran it. One successful run in a handful of runs is noise. The challenge of creating a hard task is getting here without losing realism or fairness.

Tricks like making your prompt vague, setting stricter graders, and over weighting one criterion don’t actually make your task harder. These just make the task less realistic or less fair, instead of improving the task to be a hard, real-life task. If current models can easily solve your task, you need to find a harder problem instead of increasing complexity in the grader.

Real difficulty should come from work inside the environment itself. In a coding task the answer might be spread across fixing multiple features that depend on each other, across source files, configs, and tests. In a browser task, you might need to navigate multiple pages, compare conflicting content, and keep track of changes along the way.

In one task, we gave the agent results of 10 runs of a mixture-of-experts model, along with code, configs, past run records, and metrics. We asked the agent to work out what the experiments showed. At first the runs looked successful because training loss improved every time, but the routing metrics, which show which experts actually handled the tokens, showed us that most tokens were going to only a few experts, and most of the mixture was going unused.

To summarize what happened correctly, the agent had to notice that conflict, separate the main experiments from the irrelevant runs, and figure out what actually happened.

Running your task is the first step to figuring out what a hard task is. As capabilities improve with every new model release, what was hard 6 months ago is no longer hard now. The only way to keep up is to keep building and running more tasks and watching where they break.

## Getting started

In your everyday work there’s always content you can make tasks out of. A bug report that someone files, a codebase with bugs to fix, a daily workflow or CI/CD that’s used on a daily basis. Start with one of these, write down what a successful outcome looks like, then figure out ways to check for correctness. That’s how you start.

You can run your task against any model and read the trajectories, especially the ones that seem to score perfectly. A perfect score may not mean that your task is a working, easy to solve task. When you look at the path taken by a model to solve your problem in the first pass, you might find that it took a shortcut you didn’t account for, accessed files it shouldn’t have access to, or was incorrectly scored as passing by a weak grader.

QA agents can help here. We have these on HUD, where they read a frozen snapshot of every single attempt (prompt, steps, tool calls, logs, reward) so you can see exactly where things went wrong. You can [start from a template](https://www.hud.ai/environments/new) on [hud.ai](https://hud.ai), or build an environment yourself using our [SDK documentation](https://docs.hud.ai/v6/start).
