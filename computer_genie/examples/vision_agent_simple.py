            except ActionFailedError as e:
                print(f\"Get action failed: {e}.\")

            try:
                await agent.act(\"Complete the form\")
            except ActionFailedError as e:
                print(f\"Act action failed: {e}.\")

        return 0
    except KeyboardInterrupt:
        print(\"Interrupted by user.\")
        return 130
    except Exception as e:
        print(f\"Unexpected error: {e}\")
        return 1

if __name__ == \"__main__\":
