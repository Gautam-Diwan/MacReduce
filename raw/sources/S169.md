

# Amazon EC2 Mac instances
<a name="ec2-mac-instances"></a>

EC2 Mac instances are ideal for developing, building, testing, and signing applications for Apple platforms, such as iPhone, iPad, Mac, Vision Pro, Apple Watch, Apple TV, and Safari. You can connect to your Mac instance using SSH or Apple Remote Desktop (ARD).

**Note**  
The **unit of billing** is the **dedicated host**. The instances running on that host have no additional charge.

Amazon EC2 Mac instances natively support the macOS operating system.
+ **EC2 x86 Mac instances** (`mac1.metal`) are built on 2018 Mac mini hardware powered by 3.2 GHz Intel eighth-generation (Coffee Lake) Core i7 processors, 6 physical and 12 logical cores, and 32 GiB of memory.
+ **EC2 M1 Mac instances** (`mac2.metal`) are built on 2020 Mac mini hardware powered by Apple silicon M1 processor, 8 CPU cores, 8 GPU cores, 16 GiB of memory, and the 16-core Apple Neural Engine.
+ **EC2 M1 Ultra Mac instances** (`mac2-m1ultra.metal`) are built on 2022 Mac Studio hardware powered by Apple silicon M1 Ultra processor, 20 CPU cores, 64 GPU cores, 128 GiB of memory, and the 32-core Apple Neural Engine.
+ **EC2 M2 Mac instances** (`mac2-m2.metal`) are built on 2023 Mac mini hardware powered by Apple silicon M2 processor, 8 CPU cores, 10 GPU cores, 24 GiB of memory, and the 16-core Apple Neural Engine.
+ **EC2 M2 Pro Mac instances** (`mac2-m2pro.metal`) are built on 2023 Mac mini hardware powered by Apple silicon M2 Pro processor, 12 CPU cores, 19 GPU cores, 32 GiB of memory, and the 16-core Apple Neural Engine.
+ **EC2 M4 Mac instances** (`mac-m4.metal`) are built on 2024 Mac mini hardware powered by Apple silicon M4 processor, 10 CPU cores, 10 GPU cores, 24 GiB of memory, and the 16-core Apple Neural Engine.
+ **EC2 M4 Pro Mac Mac instances** (`mac-m4pro.metal`) are built on 2024 Mac mini hardware powered by Apple silicon M4 Pro processor, 14 CPU cores, 20 GPU cores, 48 GiB of memory, and the 16-core Apple Neural Engine.

Amazon EC2 Mac Dedicated Hosts support [Dedicated Host auto recovery](dedicated-hosts-recovery.md) and [reboot-based host maintenance](dedicated-hosts-maintenance.md).

**Topics**
+ [Considerations](#mac-instance-considerations)
+ [Instance readiness](#mac-instance-readiness)
+ [EC2 macOS AMIs](#ec2-macos-images)
+ [EC2 macOS Init](#ec2-macos-init)
+ [Amazon EC2 System Monitor for macOS](#mac-instance-system-monitor)
+ [Related resources](#related-resources)
+ [Launch a Mac instance using the AWS Management Console or the AWS CLI](mac-instance-launch.md)
+ [Connect to your Mac instance using SSH or a GUI](connect-to-mac-instance.md)
+ [Update the operating system and software on Amazon EC2 Mac instances](mac-instance-updates.md)
+ [Increase the size of an EBS volume on your Mac instance](mac-instance-increase-volume.md)
+ [Stop or terminate your Amazon EC2 Mac instance](mac-instance-stop.md)
+ [Configure System Integrity Protection for Amazon EC2 Mac instances](mac-sip-settings.md)
+ [Find supported macOS versions for your Amazon EC2 Mac Dedicated Host](macos-firmware-visibility.md)
+ [Subscribe to macOS AMI notifications](macos-subscribe-notifications.md)
+ [Retrieve macOS AMI IDs using AWS Systems Manager Parameter Store API](macos-ami-ids-parameter-store.md)
+ [Amazon EC2 macOS AMIs release notes](macos-ami-overview.md)

## Considerations
<a name="mac-instance-considerations"></a>

The following considerations apply to Mac instances:
+ Mac instances are available only as bare metal instances on [Dedicated Hosts](dedicated-hosts-overview.md), with a minimum allocation period of 24 hours before you can release the Dedicated Host. You can launch one Mac instance per Dedicated Host. You can share the Dedicated Host with the AWS accounts or organizational units within your AWS organization, or the entire AWS organization.
+ Mac instances are available in different AWS Regions. For a list of Mac instance availability in each of the AWS Regions, see [Amazon EC2 instance types by Region](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html).
+ Mac instances are available only as On-Demand Instances. They are not available as Spot Instances or Reserved Instances. You can save money on Mac instances by purchasing a [Savings Plan](https://docs.aws.amazon.com/savingsplans/latest/userguide/).
+ The compatibility of different Mac instance types with specific macOS Amazon Machine Images (AMIs) varies. For more information, see [Amazon EC2 macOS AMIs release notes](macos-ami-overview.md).
+ EBS hotplug is supported.
+ AWS does not manage or support the internal SSD on the Apple hardware. We strongly recommend that you use Amazon EBS volumes instead. EBS volumes provide the same elasticity, availability, and durability benefits on Mac instances as they do on any other EC2 instance.
+ We recommend using an Amazon EBS volume with 10,000 IOPS and 400 MiB/s throughput with Mac instances for optimal performance. For more information, see [Amazon EBS volume types](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-volume-types.html) in the *Amazon EBS User Guide*.
+ [Mac instances support Amazon EC2 Auto Scaling.](https://aws.amazon.com/blogs/compute/implementing-autoscaling-for-ec2-mac-instances/) 
+ On x86 Mac instances, automatic software updates are disabled. We recommend that you apply updates and test them on your instance before you put the instance into production. For more information, see [Update the operating system and software on Amazon EC2 Mac instances](mac-instance-updates.md).
+ When you stop or terminate a Mac instance, a scrubbing workflow is performed on the Dedicated Host. For more information, see [Stop or terminate your Amazon EC2 Mac instance](mac-instance-stop.md).
+ 
**Important**  
Apple Intelligence features are not available when booting Mac hardware from an external volume. As EC2 Mac instances boot from external EBS volumes by default, they do not support Apple Intelligence features.
+ 
**Warning**  
Do not use FileVault. If you enable FileVault, the host fails to boot because the partitions are locked. If data encryption is required, use Amazon EBS encryption to avoid boot issues and performance impact. With Amazon EBS encryption, encryption operations occur on the host servers, ensuring the security of both data-at-rest and data-in-transit between the instances and their attached EBS storage. For more information, see [Amazon EBS encryption](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-encryption.html) in the *Amazon EBS User Guide*.

## Instance readiness
<a name="mac-instance-readiness"></a>

After you launch a Mac instance, you'll need to wait until the instance is ready before you can connect to it. For an AWS vended AMI with a x86 Mac instance or a Apple silicon Mac instance, the launch time can range from approximately 6 minutes to 20 minutes. Depending on the chosen Amazon EBS volume sizes, the inclusion of additional scripts to *user data*, or additional loaded software on a custom macOS AMI, the launch time might increase.

You can use a small shell script, like the one below, to poll the describe-instance-status API to know when the instance is ready to be connected to. In the following command, replace the example instance ID with your own.

```
for i in $(seq 1 200); do aws ec2 describe-instance-status --instance-ids={{i-1234567890abcdef0}} \
    --query='InstanceStatuses[0].InstanceStatus.Status'; sleep 5; done;
```

## EC2 macOS AMIs
<a name="ec2-macos-images"></a>

Amazon EC2 macOS is designed to provide a stable, secure, and high-performance environment for developer workloads running on Amazon EC2 Mac instances. EC2 macOS AMIs includes packages that enable easy integration with AWS, such as launch configuration tools and popular AWS libraries and tools.

For more information about EC2 macOS AMIs, see [Amazon EC2 macOS AMIs release notes](macos-ami-overview.md).

AWS provides updated EC2 macOS AMIs on a regular basis that include updates to packages owned by AWS and the latest fully-tested macOS version. Additionally, AWS provides updated AMIs with the latest minor version updates or major version updates as soon as they can be fully tested and vetted. If you do not need to preserve data or customizations to your Mac instances, you can get the latest updates by launching a new instance using the current AMI and then terminating the previous instance. Otherwise, you can choose which updates to apply to your Mac instances.

For information about how to subscribe to macOS AMI notifications, see [Subscribe to macOS AMI notifications](macos-subscribe-notifications.md).

## EC2 macOS Init
<a name="ec2-macos-init"></a>

EC2 macOS Init is used to initialize EC2 Mac instances at launch. It uses priority groups to run logical groups of tasks at the same time.

The launchd plist file is `/Library/LaunchDaemons/com.amazon.ec2.macos-init.plist`. The files for EC2 macOS Init are located in `/usr/local/aws/ec2-macos-init`.

For more information, see [https://github.com/aws/ec2-macos-init](https://github.com/aws/ec2-macos-init).

## Amazon EC2 System Monitor for macOS
<a name="mac-instance-system-monitor"></a>

Amazon EC2 System Monitor for macOS provides CPU utilization metrics to Amazon CloudWatch. It sends these metrics to CloudWatch over a custom serial device in 1-minute periods. You can enable or disable this agent as follows. It is enabled by default.

```
sudo setup-ec2monitoring [enable | disable]
```

**Note**  
Amazon EC2 System Monitor for macOS is not currently supported on Apple silicon Mac instances.

## Related resources
<a name="related-resources"></a>

For information about pricing, see [Pricing](https://aws.amazon.com/ec2/instance-types/mac/#Pricing).

For more information about Mac instances, see [Amazon EC2 Mac Instances](https://aws.amazon.com/ec2/instance-types/mac/).

For more information about hardware specifications and network performance of Mac instances, see [General purpose instances](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html).