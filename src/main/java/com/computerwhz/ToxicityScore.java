package com.computerwhz;

import java.util.*;

public final class ToxicityScore {

    private final String message;
    private final Map<String, Double> scores;
    private final long computedAtEpochMillis;

    private ToxicityScore(String message, Map<String, Double> scores, Long computedAtEpochMillis) {
        this.message = Objects.requireNonNull(message, "message");
        this.scores = Collections.unmodifiableMap(new LinkedHashMap<>(scores));
        this.computedAtEpochMillis = computedAtEpochMillis != null
                ? computedAtEpochMillis
                : System.currentTimeMillis();
    }

    public static ToxicityScore of(String message, Map<String, Double> scores) {
        return new ToxicityScore(message, scores, null);
    }

    // Accessors
    public String getMessage() { return message; }
    public Map<String, Double> getScores() { return scores; }
    public long getComputedAtEpochMillis() { return computedAtEpochMillis; }

    public OptionalDouble scoreOf(String label) {
        Double v = scores.get(label);
        return (v == null) ? OptionalDouble.empty() : OptionalDouble.of(v);
    }

    // ---------- Convenience getters for all 16 labels ----------
    public OptionalDouble toxicity()                      { return scoreOf("toxicity"); }
    public OptionalDouble severeToxicity()                { return scoreOf("severe_toxicity"); }
    public OptionalDouble obscene()                       { return scoreOf("obscene"); }
    public OptionalDouble threat()                        { return scoreOf("threat"); }
    public OptionalDouble insult()                        { return scoreOf("insult"); }
    public OptionalDouble identityAttack()                { return scoreOf("identity_attack"); }
    public OptionalDouble sexualExplicit()                { return scoreOf("sexual_explicit"); }

    public OptionalDouble male()                          { return scoreOf("male"); }
    public OptionalDouble female()                        { return scoreOf("female"); }
    public OptionalDouble homosexualGayOrLesbian()        { return scoreOf("homosexual_gay_or_lesbian"); }
    public OptionalDouble christian()                     { return scoreOf("christian"); }
    public OptionalDouble jewish()                        { return scoreOf("jewish"); }
    public OptionalDouble muslim()                        { return scoreOf("muslim"); }
    public OptionalDouble black()                         { return scoreOf("black"); }
    public OptionalDouble white()                         { return scoreOf("white"); }
    public OptionalDouble psychiatricOrMentalIllness()    { return scoreOf("psychiatric_or_mental_illness"); }

    @Override public String toString() {
        return "ToxicityScore{" +
                "len=" + (message == null ? 0 : message.length()) +
                ", scores=" + scores +
                ", at=" + computedAtEpochMillis +
                '}';
    }
}
