/*
 * softmax_simple - one softmax function, one file.
 *
 * Pin D7 is HIGH only while the softmax is running, so a power meter can
 * measure the softmax and nothing else.
 *
 * Serial monitor: 115200 baud.
 */

#define MARKER_PIN 7   /* connect your power meter's trigger here */
#define N          32  /* number of inputs */
#define REPS       200 /* calls per measured window, so it lasts long
                          enough for a power meter to read */

float logits[N];
float probs[N];

/* The softmax. Subtracting the max first stops exp() from overflowing. */
void softmax(const float *in, float *out, int n)
{
    float max = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > max) {
            max = in[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = exp(in[i] - max);
        sum += out[i];
    }

    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

void setup()
{
    pinMode(MARKER_PIN, OUTPUT);
    digitalWrite(MARKER_PIN, LOW);
    Serial.begin(115200);

    /* Some test inputs. Replace with your own data. */
    for (int i = 0; i < N; i++) {
        logits[i] = (float)(i % 7) - 3.0f;
    }

    Serial.println("softmax_simple ready");
}

void loop()
{
    /* Quiet gap, so the power trace has a clear idle section before the
     * measured window. */
    Serial.flush();
    delay(250);

    digitalWrite(MARKER_PIN, HIGH);
    unsigned long t0 = micros();

    for (int r = 0; r < REPS; r++) {
        softmax(logits, probs, N);
    }

    unsigned long elapsed = micros() - t0;
    digitalWrite(MARKER_PIN, LOW);

    /* Printing happens outside the window on purpose - serial output uses
     * power too, and would otherwise be blamed on the softmax. */
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += probs[i];
    }

    Serial.print("us per softmax: ");
    Serial.print((float)elapsed / REPS, 1);
    Serial.print("   probs sum to: ");
    Serial.println(sum, 4);
}
